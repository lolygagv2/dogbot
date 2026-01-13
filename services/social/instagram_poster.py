#!/usr/bin/env python3
"""
Instagram Posting Service for WIM-Z
Auto-post photos and videos to Instagram

Features:
- Photo posting with captions
- Video/Reel posting
- Session caching for persistent login
- Hashtag management
"""

import os
import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import instagrapi
try:
    from instagrapi import Client
    from instagrapi.types import Media
    INSTAGRAPI_AVAILABLE = True
except ImportError:
    INSTAGRAPI_AVAILABLE = False
    logger.warning("instagrapi not installed. Run: pip install instagrapi")

# Session cache directory
SESSION_DIR = Path('/home/morgan/dogbot/data/instagram')
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Default hashtags for dog content
DEFAULT_HASHTAGS = [
    "#dogtraining", "#goodboy", "#doglife", "#dogsofinstagram",
    "#puppylove", "#dogstagram", "#smartdog", "#wimz", "#aidog"
]


class InstagramPoster:
    """
    Instagram posting service with session caching
    """

    def __init__(self):
        self.client: Optional[Client] = None
        self.logged_in = False
        self.username = None
        self.session_file = SESSION_DIR / "session.json"

        if not INSTAGRAPI_AVAILABLE:
            logger.error("instagrapi not available - Instagram posting disabled")
            return

        self.client = Client()

        # Try to restore previous session
        self._try_restore_session()

        logger.info("InstagramPoster initialized")

    def is_available(self) -> bool:
        """Check if Instagram posting is available"""
        return INSTAGRAPI_AVAILABLE and self.client is not None

    def is_logged_in(self) -> bool:
        """Check if currently logged in"""
        return self.logged_in

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Login to Instagram

        Args:
            username: Instagram username
            password: Instagram password

        Returns:
            Dict with login status
        """
        if not self.is_available():
            return {'success': False, 'error': 'Instagram service not available'}

        try:
            # Login
            self.client.login(username, password)
            self.username = username
            self.logged_in = True

            # Save session for future use
            self._save_session()

            logger.info(f"Logged in to Instagram as {username}")

            return {
                'success': True,
                'username': username,
                'message': 'Login successful'
            }

        except Exception as e:
            logger.error(f"Instagram login failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def logout(self) -> Dict[str, Any]:
        """Logout from Instagram"""
        if self.client and self.logged_in:
            try:
                self.client.logout()
            except:
                pass

        self.logged_in = False
        self.username = None

        # Remove session file
        if self.session_file.exists():
            self.session_file.unlink()

        return {'success': True, 'message': 'Logged out'}

    def post_photo(
        self,
        image_path: str,
        caption: str,
        add_hashtags: bool = True,
        custom_hashtags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Post a photo to Instagram

        Args:
            image_path: Path to image file
            caption: Caption text
            add_hashtags: Add default hashtags
            custom_hashtags: Custom hashtags to add

        Returns:
            Dict with post result
        """
        if not self.is_available():
            return {'success': False, 'error': 'Instagram service not available'}

        if not self.logged_in:
            return {'success': False, 'error': 'Not logged in'}

        if not os.path.exists(image_path):
            return {'success': False, 'error': f'Image not found: {image_path}'}

        try:
            # Build full caption with hashtags
            full_caption = caption

            if add_hashtags or custom_hashtags:
                tags = custom_hashtags if custom_hashtags else DEFAULT_HASHTAGS
                hashtag_str = '\n\n' + ' '.join(tags)
                full_caption = caption + hashtag_str

            # Upload photo
            media = self.client.photo_upload(
                path=image_path,
                caption=full_caption
            )

            logger.info(f"Posted photo to Instagram: {media.pk}")

            return {
                'success': True,
                'media_id': media.pk,
                'media_code': media.code,
                'url': f"https://instagram.com/p/{media.code}",
                'caption': full_caption[:100] + '...' if len(full_caption) > 100 else full_caption
            }

        except Exception as e:
            logger.error(f"Photo upload failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def post_video(
        self,
        video_path: str,
        caption: str,
        thumbnail_path: Optional[str] = None,
        add_hashtags: bool = True
    ) -> Dict[str, Any]:
        """
        Post a video/reel to Instagram

        Args:
            video_path: Path to video file
            caption: Caption text
            thumbnail_path: Optional thumbnail image
            add_hashtags: Add default hashtags

        Returns:
            Dict with post result
        """
        if not self.is_available():
            return {'success': False, 'error': 'Instagram service not available'}

        if not self.logged_in:
            return {'success': False, 'error': 'Not logged in'}

        if not os.path.exists(video_path):
            return {'success': False, 'error': f'Video not found: {video_path}'}

        try:
            # Build caption
            full_caption = caption
            if add_hashtags:
                full_caption = caption + '\n\n' + ' '.join(DEFAULT_HASHTAGS)

            # Upload as reel (short video)
            media = self.client.clip_upload(
                path=video_path,
                caption=full_caption,
                thumbnail=thumbnail_path
            )

            logger.info(f"Posted video to Instagram: {media.pk}")

            return {
                'success': True,
                'media_id': media.pk,
                'media_code': media.code,
                'url': f"https://instagram.com/reel/{media.code}",
                'caption': full_caption[:100] + '...'
            }

        except Exception as e:
            logger.error(f"Video upload failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def post_story(
        self,
        image_path: str,
        link: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Post a story to Instagram

        Args:
            image_path: Path to image file
            link: Optional link to add to story

        Returns:
            Dict with post result
        """
        if not self.is_available():
            return {'success': False, 'error': 'Instagram service not available'}

        if not self.logged_in:
            return {'success': False, 'error': 'Not logged in'}

        try:
            # Upload story
            if link:
                media = self.client.photo_upload_to_story(
                    path=image_path,
                    links=[{"webUri": link}]
                )
            else:
                media = self.client.photo_upload_to_story(path=image_path)

            logger.info(f"Posted story to Instagram: {media.pk}")

            return {
                'success': True,
                'media_id': media.pk
            }

        except Exception as e:
            logger.error(f"Story upload failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_account_info(self) -> Dict[str, Any]:
        """Get current account info"""
        if not self.logged_in:
            return {'logged_in': False}

        try:
            user_info = self.client.account_info()
            return {
                'logged_in': True,
                'username': user_info.username,
                'full_name': user_info.full_name,
                'followers': user_info.follower_count,
                'following': user_info.following_count,
                'posts': user_info.media_count
            }
        except Exception as e:
            return {
                'logged_in': True,
                'username': self.username,
                'error': str(e)
            }

    def _save_session(self):
        """Save session to file for persistence"""
        if not self.client:
            return

        try:
            session_data = self.client.get_settings()
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f)
            logger.info("Instagram session saved")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def _try_restore_session(self):
        """Try to restore previous session"""
        if not self.session_file.exists():
            return

        try:
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)

            self.client.set_settings(session_data)

            # Verify session is still valid
            self.client.get_timeline_feed()

            self.logged_in = True
            self.username = session_data.get('username')
            logger.info(f"Restored Instagram session for {self.username}")

        except Exception as e:
            logger.warning(f"Failed to restore session: {e}")
            self.logged_in = False
            # Remove invalid session
            self.session_file.unlink()


# Singleton
_instagram_poster = None


def get_instagram_poster() -> InstagramPoster:
    """Get or create InstagramPoster instance"""
    global _instagram_poster
    if _instagram_poster is None:
        _instagram_poster = InstagramPoster()
    return _instagram_poster


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    poster = InstagramPoster()
    print(f"Instagram service available: {poster.is_available()}")
    print(f"Logged in: {poster.is_logged_in()}")

    # Test login (uncomment with real credentials)
    # result = poster.login("username", "password")
    # print(f"Login result: {result}")
