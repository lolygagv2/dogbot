#!/usr/bin/env python3
"""
LLM Service for WIM-Z
OpenAI GPT-4 integration for captions, summaries, and personality insights

Features:
- Photo captions (GPT-4 Vision)
- Weekly summary narratives
- Dog personality insights
- Cost-optimized with caching
"""

import os
import base64
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. Run: pip install openai")

# Load API key from environment or .env file
def _load_api_key() -> Optional[str]:
    """Load OpenAI API key from environment or .env file"""
    # Check environment first
    key = os.environ.get('OPENAI_API_KEY')
    if key:
        return key

    # Try .env file
    env_path = Path('/home/morgan/dogbot/.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    return line.split('=', 1)[1].strip()

    return None


class LLMService:
    """
    LLM Service for generating captions and summaries
    """

    def __init__(self):
        self.api_key = _load_api_key()
        self.client = None

        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("LLM Service initialized with OpenAI")
        else:
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI library not available")
            if not self.api_key:
                logger.warning("OpenAI API key not found")

        # Simple response cache to reduce API calls
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 3600  # 1 hour

    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.client is not None

    def generate_photo_caption(
        self,
        image_path: str,
        dog_name: Optional[str] = None,
        context: Optional[str] = None,
        style: str = "friendly"
    ) -> Dict[str, Any]:
        """
        Generate a caption for a dog photo using GPT-4 Vision

        Args:
            image_path: Path to the image file
            dog_name: Name of the dog (optional)
            context: Additional context (e.g., "just learned to sit")
            style: Caption style (friendly, funny, inspirational, hashtag)

        Returns:
            Dict with caption and metadata
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'LLM service not available',
                'caption': self._fallback_caption(dog_name, context)
            }

        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # Determine image type
            ext = Path(image_path).suffix.lower()
            media_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'

            # Build prompt
            style_guides = {
                'friendly': "Write a warm, friendly caption that dog lovers would enjoy.",
                'funny': "Write a humorous, witty caption that will make people smile.",
                'inspirational': "Write an uplifting, heartwarming caption about the bond between dogs and humans.",
                'hashtag': "Write a short caption followed by relevant Instagram hashtags."
            }

            prompt = f"""You are a social media caption writer for a dog training app called WIM-Z.

Look at this photo and write a short, engaging caption (1-2 sentences max).

{style_guides.get(style, style_guides['friendly'])}

{"The dog's name is " + dog_name + "." if dog_name else ""}
{"Context: " + context if context else ""}

Keep it natural and authentic. Don't be overly enthusiastic or use too many exclamation marks.
Just write the caption, nothing else."""

            # Call GPT-4 Vision
            response = self.client.chat.completions.create(
                model="gpt-4o",  # GPT-4 Vision
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}",
                                    "detail": "low"  # Use low detail to reduce cost
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150
            )

            caption = response.choices[0].message.content.strip()

            # Get usage info for cost tracking
            usage = response.usage
            estimated_cost = (usage.prompt_tokens * 0.01 + usage.completion_tokens * 0.03) / 1000

            logger.info(f"Generated caption for {image_path} (cost: ${estimated_cost:.4f})")

            return {
                'success': True,
                'caption': caption,
                'dog_name': dog_name,
                'style': style,
                'tokens_used': usage.total_tokens,
                'estimated_cost': estimated_cost
            }

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'caption': self._fallback_caption(dog_name, context)
            }

    def generate_weekly_narrative(
        self,
        report_data: Dict[str, Any],
        dog_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a narrative summary of weekly stats

        Args:
            report_data: Weekly report data from WeeklySummary
            dog_name: Focus on specific dog (optional)

        Returns:
            Dict with narrative text
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'LLM service not available',
                'narrative': self._fallback_narrative(report_data)
            }

        try:
            # Build context from report data
            context = f"""Weekly Report Data:
- Total barks: {report_data.get('bark_stats', {}).get('total_barks', 0)}
- Total treats: {report_data.get('reward_stats', {}).get('total_treats', 0)}
- Coaching sessions: {report_data.get('coaching', {}).get('total_sessions', 0)}
- Success rate: {report_data.get('coaching', {}).get('success_rate', 0)}%
- Silent Guardian effectiveness: {report_data.get('silent_guardian', {}).get('effectiveness_rate', 0)}%

Dog summary: {report_data.get('dog_summary', {})}
Highlights: {report_data.get('highlights', [])}"""

            prompt = f"""You are writing a friendly weekly summary for a dog owner using WIM-Z, an AI dog training robot.

Based on this data, write a 2-3 paragraph summary of how the dogs did this week. Be encouraging but honest.
Focus on progress, achievements, and areas for improvement.

{context}

{"Focus especially on " + dog_name + "'s progress." if dog_name else ""}

Write in a warm, supportive tone like a trainer giving feedback to a pet parent.
Don't just list stats - tell a story about the week."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for text-only tasks
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )

            narrative = response.choices[0].message.content.strip()

            usage = response.usage
            estimated_cost = (usage.prompt_tokens * 0.00015 + usage.completion_tokens * 0.0006) / 1000

            logger.info(f"Generated weekly narrative (cost: ${estimated_cost:.4f})")

            return {
                'success': True,
                'narrative': narrative,
                'tokens_used': usage.total_tokens,
                'estimated_cost': estimated_cost
            }

        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'narrative': self._fallback_narrative(report_data)
            }

    def generate_dog_personality(
        self,
        dog_name: str,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a personality description for a dog based on their behavior data

        Args:
            dog_name: Dog's name
            stats: Behavioral statistics

        Returns:
            Dict with personality text
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'LLM service not available',
                'personality': f"{dog_name} is a wonderful companion learning new things every day!"
            }

        try:
            prompt = f"""Based on this behavioral data for a dog named {dog_name}, write a short (2-3 sentences)
personality description that a pet owner would enjoy reading on their dog's profile page.

Data:
- Barks this month: {stats.get('barks', 0)}
- Treats earned: {stats.get('treats', 0)}
- Training sessions: {stats.get('sessions', 0)}
- Tricks completed: {stats.get('tricks_completed', 0)}
- Success rate: {stats.get('success_rate', 0)}%

Make it sound like a horoscope or personality profile - fun and insightful but based on the actual data.
Be creative but keep it positive."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )

            personality = response.choices[0].message.content.strip()

            return {
                'success': True,
                'personality': personality,
                'dog_name': dog_name
            }

        except Exception as e:
            logger.error(f"Personality generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'personality': f"{dog_name} is a wonderful companion learning new things every day!"
            }

    def _fallback_caption(self, dog_name: Optional[str], context: Optional[str]) -> str:
        """Generate fallback caption without LLM"""
        if dog_name and context:
            return f"{dog_name} {context}. What a good pup!"
        elif dog_name:
            return f"{dog_name} looking adorable as always!"
        elif context:
            return f"Training progress: {context}"
        else:
            return "Another great day of training!"

    def _fallback_narrative(self, report_data: Dict) -> str:
        """Generate fallback narrative without LLM"""
        barks = report_data.get('bark_stats', {}).get('total_barks', 0)
        treats = report_data.get('reward_stats', {}).get('total_treats', 0)
        sessions = report_data.get('coaching', {}).get('total_sessions', 0)

        return f"""This week saw {barks} barks detected and {treats} treats earned across {sessions} training sessions.
Keep up the great work with your furry friends!"""


# Singleton
_llm_service = None


def get_llm_service() -> LLMService:
    """Get or create LLMService instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    service = LLMService()
    print(f"LLM Service available: {service.is_available()}")

    if service.is_available():
        # Test caption generation
        test_path = "/home/morgan/dogbot/captures/snapshot_20260110_171926.jpg"
        if os.path.exists(test_path):
            result = service.generate_photo_caption(
                test_path,
                dog_name="Elsa",
                context="just learned to sit",
                style="friendly"
            )
            print(f"Caption result: {result}")
