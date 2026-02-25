import json
import os
import structlog
import time
from typing import Dict, Any, List, Optional

logger = structlog.get_logger(__name__)

class SpecialistRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpecialistRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.config_path = os.path.join(os.path.dirname(__file__), "specialists.json")
        self.specialists: Dict[str, Any] = {}
        self.mapping: Dict[str, str] = {}
        self.last_loaded = 0
        self.load_registry()
        self._initialized = True

    def load_registry(self):
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"registry_file_missing", path=self.config_path)
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.specialists = data.get("specialists", {})
            self._build_mapping()
            self.last_loaded = time.time()
            logger.info("specialist_registry_loaded", count=len(self.specialists))
        except Exception as e:
            logger.error("registry_load_failed", error=str(e))

    def _build_mapping(self):
        """Build flat mapping from aliases to canonical name"""
        self.mapping = {}
        for canonical, config in self.specialists.items():
            # Map canonical to itself
            self.mapping[canonical] = canonical
            # Map name_cn to canonical
            if "name_cn" in config:
                 self.mapping[config["name_cn"]] = canonical
            # Map aliases
            for alias in config.get("aliases", []):
                self.mapping[alias] = canonical

    def get_specialist_config(self, department: str) -> Optional[Dict[str, Any]]:
        # Try direct match
        if department in self.specialists:
            return self.specialists[department]
        
        # Try mapping
        canonical = self.mapping.get(department)
        if canonical and canonical in self.specialists:
            return self.specialists[canonical]
            
        return None

    def get_mapping(self) -> Dict[str, str]:
        return self.mapping
        
    def get_all_specialists(self) -> Dict[str, Any]:
        return self.specialists

    def reload(self):
        logger.info("reloading_registry")
        self.load_registry()

# Global instance
registry = SpecialistRegistry()
