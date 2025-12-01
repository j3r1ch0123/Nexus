#!/usr/bin/env python3.11

# Standard library imports
import json
import threading
import logging
import time
import re
import os
import io
import queue
import subprocess
import sys
import csv
import zipfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List
import geocoder
from geopy.geocoders import Nominatim
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

# Third-party imports
import speech_recognition as sr
import pyttsx3
import faiss
import torch
import numpy as np
import sounddevice as sd
from deep_translator import GoogleTranslator
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from datasets import Dataset, load_dataset as hf_load_dataset

# Local imports
from database import ResearchDatabase

class Config:
    """Configuration loader for Nexus AI assistant."""

    def __init__(self, path="config.json"):
        # Initialize database
        self.db = ResearchDatabase()
        self.db.migrate()

        # Defaults
        self._load_defaults()

        # Load JSON config
        try:
            with open(path, "r") as f:
                config = json.load(f) if config else self._load_defaults()
                with open(path, "w") as f:
                    json.dump(config, f, indent=2)

            # Optional overrides

            self.SPEECH_TIMEOUT = config.get("SPEECH_TIMEOUT", self.SPEECH_TIMEOUT)
            self.PHRASE_TIME_LIMIT = config.get("PHRASE_TIME_LIMIT", self.PHRASE_TIME_LIMIT)

            # HOTWORDS should always be a list
            hotwords = config.get("HOTWORDS")
            if isinstance(hotwords, dict):
                # Flatten dict to list
                self.HOTWORDS = [hw for kws in hotwords.values() for hw in kws]
            elif isinstance(hotwords, list):
                self.HOTWORDS = hotwords
            else:
                self.HOTWORDS = self.HOTWORDS  # fallback

            # Supported models
            self.SUPPORTED_MODELS = config.get("SUPPORTED_MODELS", self.SUPPORTED_MODELS)

            # Optional API keys
            self.API_KEYS = config.get("API_KEYS", {"NEWSAPI": os.getenv("NEWSAPI_KEY"), "WIKIPEDIA": os.getenv("WIKIPEDIA_KEY")})

            # Optional Database config
            self.DATABASE = config.get(
                "DATABASE",
                {"PATH": "research.db", "TABLE": "research_results"}
            )

        except Exception as e:
            print(f"[Config] Error loading {path}: {e}")
            print("[Config] Using defaults")

    def _load_defaults(self):
        """Fallback default configuration."""
        self.SPEECH_TIMEOUT = 5
        self.PHRASE_TIME_LIMIT = 10
        self.HOTWORDS = ["Nexus", "hey Nexus", "Nexus are you there", "Nexus wake up"]
        self.SUPPORTED_MODELS = {
        "EleutherAI/gpt-neo-125M": "EleutherAI/gpt-neo-125M",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "EleutherAI/gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B"
        }


        self.API_KEYS = {"NEWSAPI": os.getenv("NEWSAPI_KEY"), "WIKIPEDIA": os.getenv("WIKIPEDIA_KEY")}
        self.DATABASE = {"PATH": "research.db", "TABLE": "research_results"}

config = Config()

@dataclass
class AIResponse:
    """Data class to hold AI response information."""
    text: str
    status_code: int = 200

class Nexus:
    def __init__(self, config):
        self.config = config
        self.current_model = {}
        self.listening = False
        self.hotwords = config.HOTWORDS
        self.conversation_history = []
        self.tts_engine = pyttsx3.init()
        self.models = {}

        # Initialize TTS pipeline
        try:
            self.model_pipeline = self._init_tts_pipeline()
        except Exception as e:
            logging.error(f"TTS init failed: {e}")
            print(f"TTS init failed: {e}")
            self.model_pipeline = None

        self.name = "Jericho" # Change this
        self.style = (
            "formal, articulate, calmly intelligent, subtly witty, helpful, "
            "and unfailingly polite — similar to JARVIS from Iron Man."
        )
        self.system_prompt = (
            "You are Nexus, an advanced personal AI assistant inspired by JARVIS "
            "from the Iron Man films. You are polite, precise, efficient, "
            "and subtly witty. You speak with calm confidence. You anticipate "
            "the user's needs and provide clear, intelligent responses. "
            "Do NOT generate long monologues; keep replies short, elegant, and purposeful."
            "Keep your responses concise, direct, and below 100 words."
        )
        self.temperature = 0.55

    def _init_tts_pipeline(self):
        """Initialize offline TTS engine (pyttsx3)."""
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        return engine

    def generate_greeting(self):
        greeting = "Hello, I am Nexus. How may I assist you today?"
        return greeting

    def speak(self, text: str) -> None:
        """Speak text using pyttsx3."""
        if isinstance(text, AIResponse):
            text = text.text  # extract string from AIResponse
        if not hasattr(self, 'tts_engine') or self.tts_engine is None:
            print("TTS engine not initialized.")
            return

        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
            logging.error(f"TTS error: {e}")

    def hotword_detection(self, hotwords: List[str]):
        """Continuously listen for hotwords."""
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        print("Hotword detection started...")
        while self.listening:
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    text = recognizer.recognize_google(audio).lower()
                    for hw in hotwords:
                        if hw.lower() in text:
                            print(f"Hotword detected: {hw}")
                            self.speak("Yes?")
                            self.talk()  # Start conversation
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Recognizer request error: {e}")
            except Exception as e:
                print(f"Hotword detection error: {e}")

    def start_listening(self):
        """Start listening thread for hotwords."""
        self.listening = True
        threading.Thread(target=self.hotword_detection, args=(self.hotwords,), daemon=True).start()

    def stop_listening(self):
        """Stop hotword detection."""
        self.listening = False

    def talk(self):
        """Capture user speech, respond, and TTS reply."""
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        print("Say 'exit' to stop the conversation.")

        while True:
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source)
                    print("Listening...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    text = recognizer.recognize_google(audio)
                    print(f"You said: {text}")

                    if "exit" in text.lower():
                        print("Ending conversation.")
                        break

                    self.conversation_history.append({"role": "user", "content": text})

                    # For now, simple echo response (replace with your model call)
                    response = f"You said: {text}"
                    self.conversation_history.append({"role": "assistant", "content": response})
                    print(f"Nexus: {response}")
                    self.speak(response)

            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Recognizer request error: {e}")
            except KeyboardInterrupt:
                print("Conversation stopped by user.")
                break
            except Exception as e:
                print(f"Talk error: {e}")
                break

    def select_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        self.tokenizer = tokenizer
        self.model = model

    # ==================== Logging ====================
    def _setup_logging(self) -> None:
        logging.basicConfig(
            filename="./Nexus.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info("Nexus initialized.")

    # ==================== Context ====================
    def update_context(self, relationship: str, value: str) -> None:
        self.relationship_context[relationship] = value

    def get_context(self, relationship: str) -> str:
        return self.relationship_context.get(relationship, "")

    # ==================== Model Management ====================
    def load_model(self, model_name: str):
        """Load a Transformers model locally if not already loaded."""
        if model_name not in self.models:
            try:
                print(f"Loading model: {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                self.models[model_name] = (tokenizer, model)
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return None
        return self.models[model_name]

    # ==================== AI Response ====================
    def generate_response(self, prompt: str, max_length: int = 200, model_name: str = "EleutherAI/gpt-neo-125M") -> AIResponse:
        """Generate AI response using local Transformers model."""
        if not self.current_model:
            self.current_model = model_name

        model_data = self.load_model(self.current_model)
        if not model_data:
            return AIResponse(f"Failed to load model {self.current_model}", status_code=500)

        tokenizer, model = model_data
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": text})
            return AIResponse(text)
        except Exception as e:
            print(f"Error generating response: {e}")
            return AIResponse("Error generating response", status_code=500)

    # ==================== Conversation Management ====================
    def clear_conversation(self) -> None:
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        print("Conversation history cleared.")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def save_conversation(self, filename: str = "conversation_history.json") -> None:
        try:
            with open(filename, "w") as f:
                json.dump(self.conversation_history, f, indent=2)
            print(f"Conversation saved to {filename}")
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def load_conversation(self, filename: str = "conversation_history.json") -> None:
        try:
            with open(filename, "r") as f:
                self.conversation_history = json.load(f)
            print(f"Conversation loaded from {filename}")
        except Exception as e:
            print(f"Error loading conversation: {e}")

    # ==================== Hotword Translation ====================
    def translate_hotwords(self, hotwords: List[str], target_languages: List[str] = None) -> List[str]:
        if target_languages is None:
            target_languages = ["es", "fr"]
        translator = GoogleTranslator()
        translated_hotwords: List[str] = []
        for lang in target_languages:
            for hw in hotwords:
                try:
                    translated = translator.translate(hw, target=lang)
                    translated_hotwords.append(translated)
                except Exception:
                    translated_hotwords.append(hw)
        return translated_hotwords

class DeepResearch:
    def __init__(self):
        self.search_engines = ['google', 'wikipedia', 'news']
        self.max_results = 5
        self.newsapi = newsapi.NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
        self.wiki = wikipediaapi.Wikipedia('en')
        
    def web_search(self, query):
        results = []
        results.extend(self._google_search(query))
        results.extend(self._wikipedia_search(query))
        results.extend(self._news_search(query))
        return results
        
    def _google_search(self, query):
        try:
            for url in search(query, num=self.max_results, stop=self.max_results, pause=2):
                results.append(self._process_url(url))
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
            
    def _process_url(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            return {
                'title': soup.title.string if soup.title else 'Untitled',
                'url': url,
                'content': soup.get_text()[:500] + '...'
            }
        except Exception as e:
            print(f"URL processing error: {e}")
            return None

    def _wikipedia_search(self, query):
        page = self.wiki.page(query)
        if page.exists():
            return [{
                'title': page.title,
                'url': page.fullurl,
                'content': page.summary[:500] + '...',
                'source': 'Wikipedia'
            }]
        return []
        
    def _news_search(self, query):
        articles = self.newsapi.get_everything(q=query, language='en', page_size=self.max_results)
        return [{
            'title': article['title'],
            'url': article['url'],
            'content': article['description'][:500] + '...',
            'source': article['source']['name'],
            'published_at': article['publishedAt']
        } for article in articles['articles']]
        
    def save_results(self, results, format='json'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if format == 'json':
            with open(f'research_results_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=2)
        elif format == 'md':
            with open(f'research_results_{timestamp}.md', 'w') as f:
                for result in results:
                    f.write(f"## {result['title']}\n")
                    f.write(f"**Source:** {result['source']}\n")
                    f.write(f"**URL:** {result['url']}\n")
                    f.write(f"{result['content']}\n\n")
    
    def start(self, query):
        while True:
            results = self.web_search(query)
            self.save_results(results)
            print(f"Results saved to research_results_{timestamp}.json and research_results_{timestamp}.md")
            print("Do you want to search again? (y/n)")
            if input().lower() != 'y':
                break

        print("Goodbye!")

class FineTuner:
    def __init__(self, index_path="vector.index", meta_path="meta.json", model_name="gpt2", embed_model_name="all-MiniLM-L6-v2"):
        self.all_texts = []
        self.index_path = index_path
        self.meta_path = meta_path
        self.memory_meta = []

        # FAISS index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(384)  # <-- 384 for MiniLM, adjust if you pick a different model

        # Metadata
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.memory_meta = json.load(f)

        # Fine-tuning model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # **Embedding model**
        self.memory_encoder = SentenceTransformer(embed_model_name)

    # --- File Loading ---
    def choose_dataset(self):
        # Ask user to select training files
        file_paths = filedialog.askopenfilenames(title="Select Training Files")
        if not file_paths:
            return

        # Extract text from each file
        for path in file_paths:
            try:
                self.extract_text(path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed reading {path}:\n{e}")

        if not self.all_texts:
            messagebox.showwarning("Nexus", "No valid text extracted.")
            return

        # Generate embeddings
        try:
            embeddings = self.encode_texts(self.all_texts)

            # Convert to NumPy array (FAISS requires float32)
            embeddings = np.array(embeddings, dtype=np.float32)

            # Ensure correct shape
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)  # single vector
            elif embeddings.ndim != 2:
                raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

            # Add to FAISS index
            self.index.add(embeddings)

        except Exception as e:
            messagebox.showerror("Error", f"Embedding failure:\n{e}")
            return

        # Extend metadata
        for t in self.all_texts:
            self.memory_meta.append({"text": t})

        # Save updated FAISS index and metadata
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.memory_meta, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed saving memory:\n{e}")
            return

        messagebox.showinfo("Nexus", "Training data added to vector memory!")

    def extract_text(self, path):
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                self.all_texts.append(f.read())
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.flatten_json(data)
        elif path.endswith(".csv"):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.extract_row_text(row)
        elif path.endswith(".zip"):
            with zipfile.ZipFile(path, 'r') as zipf:
                for filename in zipf.namelist():
                    with zipf.open(filename) as f:
                        temp_path = io.BytesIO(f.read())
                        # Recursively extract
                        self.extract_text_from_bytes(temp_path, filename)

    def extract_text_from_bytes(self, byte_stream, filename):
        byte_stream.seek(0)
        if filename.endswith(".txt"):
            self.all_texts.append(byte_stream.read().decode("utf-8"))
        elif filename.endswith(".json"):
            data = json.load(byte_stream)
            self.flatten_json(data)
        elif filename.endswith(".csv"):
            reader = csv.DictReader(io.TextIOWrapper(byte_stream, encoding="utf-8"))
            for row in reader:
                self.extract_row_text(row)

    def flatten_json(self, data):
        if isinstance(data, dict):
            self.all_texts.append(json.dumps(data, indent=2))
        elif isinstance(data, list):
            for item in data:
                self.all_texts.append(json.dumps(item))
        else:
            self.all_texts.append(str(data))

    def extract_row_text(self, row):
        # Auto-detect text column
        text_column = None
        for col in row.keys():
            if col.lower() in ["text", "content", "body", "message"]:
                text_column = col
                break
        if text_column:
            self.all_texts.append(row[text_column])
        else:
            self.all_texts.append(" ".join(row.values()))

    def load_dataset(self):
        if not self.dataset_path:
            messagebox.showwarning("Nexus", "No dataset path found.")
            return

        try:
            if self.dataset_path.endswith(".json"):
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                self.dataset = [{"text": json.dumps(d)} for d in data]

            elif self.dataset_path.endswith(".csv"):
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    self.dataset = []
                    for row in reader:
                        text_col = None
                        for col in row.keys():
                            if col.lower() in ["text", "content", "body", "message"]:
                                text_col = col
                                break
                        if text_col:
                            self.dataset.append({"text": row[text_col]})
                        else:
                            self.dataset.append({"text": " ".join(row.values())})

            elif self.dataset_path.endswith(".zip"):
                self.dataset = []
                with zipfile.ZipFile(self.dataset_path, "r") as zipf:
                    for filename in zipf.namelist():
                        with zipf.open(filename) as f:
                            temp_path = io.BytesIO(f.read())
                            temp_path.seek(0)
                            if filename.endswith(".json"):
                                data = json.load(io.TextIOWrapper(temp_path, encoding='utf-8'))
                                if isinstance(data, dict):
                                    data = [data]
                                self.dataset.extend([{"text": json.dumps(d)} for d in data])
                            elif filename.endswith(".csv"):
                                reader = csv.DictReader(io.TextIOWrapper(temp_path, encoding="utf-8"))
                                for row in reader:
                                    text_col = None
                                    for col in row.keys():
                                        if col.lower() in ["text", "content", "body", "message"]:
                                            text_col = col
                                            break
                                    if text_col:
                                        self.dataset.append({"text": row[text_col]})
                                    else:
                                        self.dataset.append({"text": " ".join(row.values())})

            if not self.dataset:
                messagebox.showwarning("Nexus", "No valid data found in the selected dataset.")
            else:
                messagebox.showinfo("Nexus", f"Dataset loaded successfully! {len(self.dataset)} examples.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{e}")

    # --- Embedding ---
    def encode_texts(self, texts, batch_size=32):
        # Clean up the data
        clean_texts = [str(t) for t in texts if t]
        embeddings = []

        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i:i+batch_size]
            batch_embeddings = self.memory_encoder.encode(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    # --- Fine-tuning ---
    def fine_tune(self):
        if not self.all_texts:
            messagebox.showwarning("Nexus", "No training data loaded for fine-tuning.")
            return

        # Create prompt-completion dataset
        dataset_dict = [{"text": t, "labels": t} for t in self.all_texts]
        ds = Dataset.from_list(dataset_dict)

        def tokenize_fn(examples, tokenizer):
            tokens = tokenizer(examples["text"], truncation=True, padding="max_length")
            tokens["labels"] = tokens["input_ids"]
            return tokens

        # When mapping
        tokenized_ds = ds.map(lambda x: tokenize_fn(x, self.tokenizer), batched=True, remove_columns=["text"])

        tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        training_args = TrainingArguments(
            output_dir="./nexus_ft",
            overwrite_output_dir=True,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds
        )

        trainer.train()
        messagebox.showinfo("Nexus", "Fine-tuning completed successfully!")

class HomeAssistantControl:
    def __init__(self, weather_label=None):
        self.token = os.getenv("HOME_ASSISTANT_TOKEN")
        self.url = os.getenv("HOME_ASSISTANT_URL")
        self.home_assistant_url = self.url
        self.weather_label = weather_label  # Store a reference to the label

    def home_assistant_control(self, entity_id: str, action: str = "toggle") -> None:
        """Control a Home Assistant device.

        Args:
            entity_id: The ID of the device to control
            action: The action to perform (e.g., "toggle", "turn_on", "turn_off")
        """
        url = f"{self.home_assistant_url}/api/services/light/{action}"
        if not self.token:
            print("Home Assistant token not found. Please set the HOME_ASSISTANT_TOKEN environment variable.")
            logging.error("Home Assistant token not found. Please set the HOME_ASSISTANT_TOKEN environment variable.")
            return
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        data = {"entity_id": entity_id}

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            print(f"{action.capitalize()} {entity_id} successfully.")
            logging.info(f"{action.capitalize()} {entity_id} successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error controlling {entity_id}: {e}")
            logging.error(f"Error controlling {entity_id}: {e}")
    
    def process_home_command(self, command: str) -> None:
        """Process a Home Assistant command.

        Args:
            command: The command to process
        """
        command = command.lower()

        # Check if the response contains a command for Home Assistant automation
        if "turn on" in command or "turn off" in command or "toggle" in command:
            entity_id = self.extract_entity_id(command)
            if entity_id:
                if "turn on" in command:
                    self.home_assistant_control(entity_id, action="turn_on")
                elif "turn off" in command:
                    self.home_assistant_control(entity_id, action="turn_off")
                else:
                    self.home_assistant_control(entity_id, action="toggle")
        elif "weather" in command:
            self.handle_weather_query("weather.your_weather_entity_id")  # Replace with the actual entity ID
        elif "research" in command or "look up" in command:
            self.start_research(command)
        else:
            logging.warning(f"Unknown home command: {command}")

    def extract_entity_id(self, command: str) -> Optional[str]:
        """Extract the entity ID from a command.

        Args:
            command: The command to extract the entity ID from

        Returns:
            Optional[str]: The extracted entity ID or None if not found
        """
        entity_map = {
            "light": "light.living_room",
            "fan": "fan.ceiling_fan",
        }

        for key, entity_id in entity_map.items():
            if key in command:
                return entity_id

        print(f"Entity ID not found for command: {command}")
        logging.warning(f"Entity ID not found in command: {command}")
        return None

    def start_research(self):
        # Use the GUI to start deep research
        query = input("Enter your research query: ")
        research = DeepResearch()
        results = research.web_search(query)
        self.display_results(results)

    def display_results(self, results):
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Content: {result['content']}")
            print("\n")

    def handle_weather_query(self, entity_id: str) -> None:
        """Handle a weather query.

        Args:
            entity_id: The entity ID of the weather device
        """
        # Replace with actual weather query logic
        pass

class NexusThread:
    def __init__(self, hotwords: List[str], token: str, home_assistant_url: str):
        self.Nexus = Nexus(config=config)
        self.Nexus.interface = self
        self.log_signal = pyqtSignal(str)  # <-- add this line
        self.hotwords = hotwords
        self.token = token
        self.home_assistant_url = home_assistant_url

    def run(self) -> None:
        Nexus = Nexus()
        Nexus.home_assistant_token = self.token
        Nexus.home_assistant_url = self.home_assistant_url
        self.log_signal.emit("Initializing Nexus...")
        # Make sure the hotword detection loop is active
        Nexus.hotword_detection(hotwords=self.hotwords)
        self.log_signal.emit("Hotword detection stopped.")

class ConsoleStream:
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.update_pending = False

    def write(self, text: str) -> None:
        self.queue.put(text)
        if not self.update_pending:
            try:
                self.text_widget.after(10, self._process_queue)
                self.update_pending = True
            except tk.TclError:
                # Main loop not running yet, just ignore output
                pass

    def _process_queue(self) -> None:
        self.update_pending = False
        try:
            while True:
                text = self.queue.get_nowait()
                self.text_widget.insert(tk.END, text + "\n")
                self.text_widget.see(tk.END)
                self.queue.task_done()
        except queue.Empty:
            pass
        
    def flush(self) -> None:
        # Required by Python's IO system
        pass

class NexusInterface:
    def __init__(self):
        print("Initializing Nexus Interface...")
        nexus = Nexus(config=config)
        # Have the Nexus class generate a greeting on startup
        greeting = nexus.generate_greeting()
        nexus.speak(greeting)

        self.vector_index_path = "Nexus_memory/faiss.index"
        self.vector_meta_path = "Nexus_memory/meta.json"
        self.memory_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.memory_meta = []
        self.index = None

        # Initialize fine tuning
        self.fine_tuner = FineTuner(model_name="EleutherAI/gpt-neo-1.3B")

        # Ensure FAISS memory exists or create new
        self._load_vector_memory()

        self.window = tk.Tk()
        self.Nexus = None  # Will be set by set_Nexus
        self.window.title("Nexus Interface")
        print("Window created successfully")
        frame = tk.Frame(self.window)
        frame.pack()

        self.home = HomeAssistantControl(self.window)

        # Logo
        logo_label = tk.Label(self.window, text="Nexus", font=("Arial", 24))
        logo_label.pack(pady=10)

        # Time/date labels
        self.hour = datetime.now().strftime("%I:%M %p")
        self.time_label = tk.Label(self.window, text=f"Current Time: {self.hour}", font=("Arial", 12))
        self.time_label.pack(pady=5)
        self.update_time()

        self.date_label = tk.Label(self.window, text=f"Current Date: {datetime.now().strftime('%A, %B %d, %Y')}", font=("Arial", 12))
        self.date_label.pack(pady=5)

        # Location
        self.location_label = tk.Label(self.window, text=f"Location: {self.get_geolocation()}", font=("Arial", 12))
        self.location_label.pack(pady=5)

        # Weather
        self.weather_label = tk.Label(self.window, text="Weather: Checking...", font=("Arial", 12))
        self.weather_label.pack(pady=5)

        # Dropdown for microphones
        self.mic_var = tk.StringVar()
        mics = sr.Microphone.list_microphone_names()
        mic_frame = tk.Frame(self.window)
        mic_frame.pack(pady=5)
        tk.Label(mic_frame, text="Select Microphone:").pack(side=tk.LEFT, padx=5)
        mic_menu = ttk.Combobox(mic_frame, textvariable=self.mic_var)
        mic_menu['values'] = mics
        mic_menu.set(mics[0] if mics else "No microphones found")
        mic_menu.pack(side=tk.LEFT, padx=5)

        # Dropdown for models
        self.model_var = tk.StringVar()
        model_frame = tk.Frame(self.window)
        model_frame.pack(pady=5)
        tk.Label(model_frame, text="Select AI Model:").pack(side=tk.LEFT, padx=5)
        model_menu = ttk.Combobox(model_frame, textvariable=self.model_var)
        model_menu['values'] = list(config.SUPPORTED_MODELS.keys())
        model_menu.set(list(config.SUPPORTED_MODELS.keys())[0] if config.SUPPORTED_MODELS else "No models found")
        model_menu.pack(side=tk.LEFT, padx=5)
        tk.Button(model_frame, text="Select Model", command=self.select_model).pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Start Nexus", command=self.start_Nexus).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Stop Nexus", command=self.stop_Nexus).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Pause Nexus", command=self.pause_Nexus).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Check Weather", command=self.check_weather).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear Conversation", command=self.clear_conversation).pack(side=tk.LEFT, padx=5)
        self.research_button = tk.Button(button_frame, text="Deep Research", command=self.start_research)
        self.research_button.pack(side=tk.LEFT, padx=5)

        tk.Button(self.window, text="Load Dataset", command=self.fine_tuner.choose_dataset).pack(pady=5)
        tk.Button(self.window, text="Fine-Tune Model", command=lambda: self.fine_tuner.fine_tune()).pack(pady=5)

        # Text log
        self.text_widget = tk.Text(self.window, wrap=tk.WORD, state=tk.NORMAL)
        self.text_widget.pack(expand=True, fill=tk.BOTH, pady=10)
        scrollbar = tk.Scrollbar(self.window, command=self.text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.config(yscrollcommand=scrollbar.set)

        # Input box + send button
        self.user_input = tk.Text(self.window, height=5, width=50)
        self.user_input.pack(pady=10)
        self.send_button = tk.Button(self.window, text="Send", command=self.send_input)
        self.send_button.pack(pady=10)

        # Console redirect
        sys.stdout = ConsoleStream(self.text_widget)

        # Nexus core
        self.config = Config()  # or however you do it
        self.Nexus = Nexus(self.config)

        self.Nexus.interface = self
        self.hotwords = config.HOTWORDS

        # Start hotword detection
        self.window.after(100, self._initialize_after_mainloop)

        self.memory_encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Memory index file paths
        self.index_path = "Nexus_memory/faiss.index"
        self.meta_path = "Nexus_memory/meta.json"

        # Initialize memory
        self.memory_meta = []
        self.index = None

        self.memory_input = tk.Entry(frame)
        self.memory_input.pack(side=tk.LEFT, padx=5)

        query_button = tk.Button(
            frame,
            text="Search Memory",
            command=lambda: self.query_memory(self.memory_input.get())
        )
        query_button.pack(side=tk.LEFT, padx=5)

        # Create a button for training memory
        self.train_memory_button = tk.Button(self.window, text="Train Memory", command=self.choose_training_data)
        self.train_memory_button.pack(side=tk.LEFT, padx=5)

        # Load or initialize FAISS + metadata
        self._load_vector_memory()

    def _init_vector_memory(self):
        self.vector_index_path = "vector.index"
        self.vector_meta_path = "vector_meta.json"
        self.memory_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._load_vector_memory()

    def _load_vector_memory(self):
        directory = os.path.dirname(self.vector_index_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.exists(self.vector_index_path) and os.path.exists(self.vector_meta_path):
            try:
                self.index = faiss.read_index(self.vector_index_path)
                with open(self.vector_meta_path, "r") as f:
                    self.memory_meta = json.load(f)
                print("[Vector Memory] Loaded existing index.")
                return
            except Exception as e:
                print(f"[Vector Memory] Failed to load index, creating new: {e}")

        # Create a fresh index
        self.index = faiss.IndexFlatL2(384)
        self.memory_meta = []
        self._save_vector_memory()
        print("[Vector Memory] Created new index.")

    def _save_vector_memory(self):
        faiss.write_index(self.index, self.vector_index_path)
        with open(self.vector_meta_path, "w") as f:
            json.dump(self.memory_meta, f)

    def extract_text_from_file(self, file_obj, ext):
        if ext == '.txt':
            return [file_obj.read().decode('utf-8')]
        elif ext == '.json':
            data = json.load(io.TextIOWrapper(file_obj, 'utf-8'))
            if isinstance(data, dict):
                return [json.dumps(data, indent=2)]
            elif isinstance(data, list):
                return [json.dumps(item) for item in data]
            else:
                return [str(data)]
        elif ext == '.csv':
            reader = csv.DictReader(io.TextIOWrapper(file_obj, 'utf-8'))
            texts = []
            for row in reader:
                if 'text' in row:
                    texts.append(row['text'])
                elif 'content' in row:
                    texts.append(row['content'])
                else:
                    texts.append(" ".join(row.values()))
            return texts
        else:
            return [file_obj.read().decode('utf-8')]

    # ============================
    #  USER-SELECTED TRAINING DATA
    # ============================
    def choose_training_data(self):
        file_paths = filedialog.askopenfilenames(title="Select Training Files")
        if not file_paths:
            return

        all_texts = []

        for path in file_paths:
            try:
                if path.endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as f:
                        all_texts.append(f.read())
                elif path.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            all_texts.append(json.dumps(data, indent=2))
                        elif isinstance(data, list):
                            all_texts.extend([json.dumps(item) for item in data])
                        else:
                            all_texts.append(str(data))
                elif path.endswith(".csv"):
                    with open(path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            text_col = next((col for col in ["text", "content", "body", "message"] if col in row), None)
                            all_texts.append(str(row[text_col]) if text_col else " ".join([str(v) for v in row.values()]))
                elif path.endswith(".zip"):
                    with zipfile.ZipFile(path, 'r') as zipf:
                        for filename in zipf.namelist():
                            ext = os.path.splitext(filename)[1].lower()
                            with zipf.open(filename, 'r') as f:
                                all_texts.extend(self.extract_text_from_file(f, ext))
            except Exception as e:
                messagebox.showerror("Error", f"Failed reading {path}:\n{e}")

        all_texts = [str(t) for t in all_texts if t]  # ensure all are strings
        if not all_texts:
            messagebox.showwarning("Nexus", "No valid text extracted.")
            return

        # Batch encode to avoid memory issues
        embeddings = []
        batch_size = 32
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            emb = self.memory_encoder.encode(batch)
            embeddings.extend(emb)
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.memory_meta.extend([{"text": t} for t in all_texts])
        self._save_vector_memory()
        messagebox.showinfo("Nexus", "Training data added to vector memory!")


    # ============================
    #  MEMORY QUERY FUNCTION
    # ============================
    def query_memory(self, text: str, k=5):
        """Search Nexus's vector memory and display results in the GUI log."""
        if len(self.memory_meta) == 0:
            self.append_text("[Memory] Memory is empty.")
            return []

        # Encode the query
        emb = self.memory_encoder.encode([text]).astype("float32")
        distances, indices = self.index.search(emb, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.memory_meta):
                results.append(self.memory_meta[idx]["text"])

        # Display results in GUI text log
        if results:
            self.append_text(f"[Memory Search] Results for: '{text}'")
            for i, r in enumerate(results, start=1):
                self.append_text(f"{i}. {r}\n")
        else:
            self.append_text(f"[Memory Search] No relevant memory found for: '{text}'")
        
        # Read the results out loud with the speak function
        nexus = Nexus(config=self.config)
        for result in results:
            nexus.speak(result)
            return result

    def append_text(self, text: str) -> None:
        """Append text to the GUI text widget."""
        self.text_widget.config(state=tk.NORMAL)  # ensure editable
        self.text_widget.insert(tk.END, text + "\n")
        self.text_widget.see(tk.END)  # scroll to bottom
        self.text_widget.config(state=tk.DISABLED)  # make read-only


    def update_time(self) -> None:
        self.hour = datetime.now().strftime("%I:%M %p")
        self.time_label.config(text=f"Current Time: {self.hour}")
        self.window.after(1000, self.update_time)

    def start_research(self, query) -> None:
        # Use the deep research class
        deep_research = DeepResearch()
        deep_research.start(query)
    
    def get_geolocation(self) -> str:
        try:
            return geocoder.ip('me').country
        except Exception as e:
            print(f"Error getting geolocation: {e}")
            return "Unknown"

    def check_weather(self) -> str:
        try:
            return weather.get_weather(self.get_geolocation())
        except Exception as e:
            print(f"Error getting weather: {e}")
            return "Unknown"
    
    def _initialize_after_mainloop(self) -> None:
        threading.Thread(
            target=self.Nexus.hotword_detection,
            args=(self.Nexus.hotwords,),  # pass hotwords as a tuple
            daemon=True
        ).start()


    def run(self) -> None:
        print("Starting Nexus Interface...")
        self.window.mainloop()

    def start_Nexus(self):
        self.Nexus.listening = True

    def stop_Nexus(self) -> None:
        print("Stopping Nexus...")
        self.Nexus.listening = False
        self.window.destroy()

    def pause_Nexus(self) -> None:
        print("Pausing Nexus...")
        self.Nexus.listening = False

    def send_input(self) -> None:
        user_input = self.user_input.get("1.0", tk.END).strip()
        if not user_input:
            return

        print(f"User Input: {user_input}")

        try:
            # Select the model
            model_name = self.model_var.get()
            self.Nexus.select_model(model_name)

            pipeline = self.Nexus.model_pipeline

            if pipeline is None:
                response = "Model pipeline not initialized."

            # Case 1: HuggingFace pipeline (callable)
            elif callable(pipeline):
                out = pipeline(user_input, max_new_tokens=200)
                response = out[0]["generated_text"]

            # Case 2: Custom class with .generate()
            elif hasattr(pipeline, "generate"):
                response = pipeline.generate(user_input)

            # Case 3: Model+tokenizer pair (HF-style)
            elif hasattr(self.Nexus, "tokenizer") and hasattr(self.Nexus, "model"):
                inputs = self.Nexus.tokenizer(user_input, return_tensors="pt")
                outputs = self.Nexus.model.generate(**inputs, max_new_tokens=200)
                response = self.Nexus.tokenizer.decode(outputs[0], skip_special_tokens=True)

            else:
                response = "Unsupported model type: cannot generate text."

            # Save conversation
            self.Nexus.conversation_history.append({"role": "user", "content": user_input})
            self.Nexus.conversation_history.append({"role": "assistant", "content": response})

            # Display + speak
            self.append_text(f"You: {user_input}")
            self.append_text(f"Nexus: {response}")
            self.Nexus.speak(response)

        except Exception as e:
            print(f"Error processing user input: {e}")

        finally:
            self.user_input.delete("1.0", tk.END)


    def send_message(self, prompt: str, user_text: str, model_name: str):
        """Generate text using the selected model."""
        if not self.model_pipeline:
            print("[Nexus] Model pipeline not initialized")
            return None

        try:
            response = self.model_pipeline(user_text, max_length=200, do_sample=True)
            generated_text = response[0]['generated_text']
            # Optionally append to conversation history
            self.conversation_history.append({"role": "assistant", "content": generated_text})
            print(f"Nexus: {generated_text}")
            # Speak if TTS is initialized
            if hasattr(self, "tts") and self.tts:
                self.tts.tts_to_file(text=generated_text, file_path="output.wav")
                # Play audio using simple library, e.g., playsound
                import playsound
                playsound.playsound("output.wav")
            return 200  # mimic HTTP status
        except Exception as e:
            print(f"[Nexus] Error generating message: {e}")
            return None

    def clear_conversation(self):
        self.Nexus.conversation_history = []
        self.append_text("Conversation cleared.")

    def select_model(self):
        pipeline = self.Nexus.model_pipeline
        # Get model from dropdown
        model_name = self.model_var.get()
        try:
            self.Nexus.select_model(model_name)
            print(f"Selected model: {model_name}")
        except Exception as e:
            print(f"Error selecting model: {e}")

# Create and run the GUI
if __name__ == "__main__":
    Nexus_interface = NexusInterface()
    Nexus_interface.run()
