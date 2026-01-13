#!/usr/bin/env python3
"""
Photo Enhancement Service for WIM-Z
Auto-enhance photos and add text overlays for social media

Features:
- Auto contrast/brightness/saturation
- Text overlays (dog name, trick, date)
- Instagram-friendly sizing (1080x1080)
- Fun filters (optional)
"""

import os
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger(__name__)

# Output directory for enhanced photos
OUTPUT_DIR = Path('/home/morgan/dogbot/captures/enhanced')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Instagram dimensions
INSTAGRAM_SIZE = (1080, 1080)
INSTAGRAM_STORY_SIZE = (1080, 1920)


class PhotoEnhancer:
    """
    Photo enhancement service with auto-enhance and text overlays
    """

    def __init__(self):
        self.font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        self.font_light_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        logger.info("PhotoEnhancer initialized")

    def enhance(
        self,
        image_path: str,
        dog_name: Optional[str] = None,
        caption: Optional[str] = None,
        auto_enhance: bool = True,
        resize_instagram: bool = True,
        add_watermark: bool = True,
        filter_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance a photo with auto-adjustments and overlays

        Args:
            image_path: Path to source image
            dog_name: Dog name to overlay (optional)
            caption: Caption text to overlay (optional)
            auto_enhance: Apply auto contrast/brightness/saturation
            resize_instagram: Resize to Instagram square format
            add_watermark: Add WIM-Z watermark
            filter_name: Optional filter (warm, cool, vintage, dramatic)

        Returns:
            Dict with output_path and metadata
        """
        try:
            # Load image
            img = Image.open(image_path)
            original_size = img.size
            logger.info(f"Loaded image: {image_path} ({original_size})")

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Apply auto-enhancement
            if auto_enhance:
                img = self._auto_enhance(img)

            # Apply filter
            if filter_name:
                img = self._apply_filter(img, filter_name)

            # Resize for Instagram
            if resize_instagram:
                img = self._resize_square(img, INSTAGRAM_SIZE[0])

            # Add text overlays
            if dog_name or caption:
                img = self._add_text_overlay(img, dog_name, caption)

            # Add watermark
            if add_watermark:
                img = self._add_watermark(img)

            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dog_slug = dog_name.lower().replace(' ', '_') if dog_name else 'photo'
            output_filename = f"{dog_slug}_{timestamp}_enhanced.jpg"
            output_path = OUTPUT_DIR / output_filename

            # Save with high quality
            img.save(output_path, 'JPEG', quality=95, optimize=True)

            logger.info(f"Enhanced photo saved: {output_path}")

            return {
                'success': True,
                'output_path': str(output_path),
                'original_size': original_size,
                'output_size': img.size,
                'dog_name': dog_name,
                'caption': caption,
                'filter': filter_name,
                'timestamp': timestamp
            }

        except Exception as e:
            logger.error(f"Photo enhancement failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _auto_enhance(self, img: Image.Image) -> Image.Image:
        """Apply automatic enhancements"""
        # Auto contrast (moderate)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.15)

        # Auto brightness (slight boost)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)

        # Auto saturation (moderate boost for vibrant colors)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)

        # Slight sharpening
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)

        return img

    def _apply_filter(self, img: Image.Image, filter_name: str) -> Image.Image:
        """Apply named filter"""
        if filter_name == 'warm':
            # Warm tones - boost reds/yellows
            r, g, b = img.split()
            r = r.point(lambda x: min(255, int(x * 1.1)))
            b = b.point(lambda x: int(x * 0.9))
            img = Image.merge('RGB', (r, g, b))

        elif filter_name == 'cool':
            # Cool tones - boost blues
            r, g, b = img.split()
            r = r.point(lambda x: int(x * 0.9))
            b = b.point(lambda x: min(255, int(x * 1.1)))
            img = Image.merge('RGB', (r, g, b))

        elif filter_name == 'vintage':
            # Vintage - lower saturation, warm tint, slight vignette
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.8)
            r, g, b = img.split()
            r = r.point(lambda x: min(255, int(x * 1.05)))
            g = g.point(lambda x: int(x * 0.98))
            b = b.point(lambda x: int(x * 0.9))
            img = Image.merge('RGB', (r, g, b))

        elif filter_name == 'dramatic':
            # High contrast, slight desaturation
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.9)

        elif filter_name == 'bright':
            # Extra bright and vibrant
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.15)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.3)

        return img

    def _resize_square(self, img: Image.Image, size: int) -> Image.Image:
        """Resize to square with center crop"""
        width, height = img.size

        # Calculate crop box for center square
        if width > height:
            left = (width - height) // 2
            top = 0
            right = left + height
            bottom = height
        else:
            left = 0
            top = (height - width) // 2
            right = width
            bottom = top + width

        # Crop to square
        img = img.crop((left, top, right, bottom))

        # Resize to target size
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        return img

    def _add_text_overlay(
        self,
        img: Image.Image,
        dog_name: Optional[str] = None,
        caption: Optional[str] = None
    ) -> Image.Image:
        """Add text overlay with shadow effect"""
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Load fonts
        try:
            name_font = ImageFont.truetype(self.font_path, 72)
            caption_font = ImageFont.truetype(self.font_light_path, 36)
        except:
            name_font = ImageFont.load_default()
            caption_font = ImageFont.load_default()

        # Draw dog name (top-left with shadow)
        if dog_name:
            text = dog_name.upper()
            x, y = 40, 40

            # Shadow
            draw.text((x + 3, y + 3), text, font=name_font, fill=(0, 0, 0, 180))
            # Main text
            draw.text((x, y), text, font=name_font, fill=(255, 255, 255, 255))

        # Draw caption (bottom with shadow)
        if caption:
            # Word wrap if too long
            lines = self._wrap_text(caption, caption_font, width - 80)

            y = height - 60 - (len(lines) * 45)
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=caption_font)
                text_width = bbox[2] - bbox[0]
                x = (width - text_width) // 2

                # Shadow
                draw.text((x + 2, y + 2), line, font=caption_font, fill=(0, 0, 0, 180))
                # Main text
                draw.text((x, y), line, font=caption_font, fill=(255, 255, 255, 255))
                y += 45

        return img

    def _wrap_text(self, text: str, font: ImageFont, max_width: int) -> list:
        """Wrap text to fit within width"""
        words = text.split()
        lines = []
        current_line = []

        # Create temp draw for measuring
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)

        for word in words:
            current_line.append(word)
            line = ' '.join(current_line)
            bbox = draw.textbbox((0, 0), line, font=font)
            if bbox[2] > max_width:
                current_line.pop()
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def _add_watermark(self, img: Image.Image) -> Image.Image:
        """Add WIM-Z watermark in corner"""
        draw = ImageDraw.Draw(img)
        width, height = img.size

        try:
            font = ImageFont.truetype(self.font_light_path, 24)
        except:
            font = ImageFont.load_default()

        text = "WIM-Z"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]

        x = width - text_width - 20
        y = height - 40

        # Semi-transparent watermark
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 100))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 150))

        return img

    def create_collage(
        self,
        image_paths: list,
        dog_name: Optional[str] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a photo collage from multiple images"""
        try:
            if len(image_paths) < 2:
                return {'success': False, 'error': 'Need at least 2 images for collage'}

            # Load images
            images = [Image.open(p).convert('RGB') for p in image_paths[:4]]

            # Create 2x2 grid
            grid_size = 1080
            cell_size = grid_size // 2

            collage = Image.new('RGB', (grid_size, grid_size), (30, 30, 30))

            positions = [(0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size)]

            for i, img in enumerate(images):
                if i >= 4:
                    break
                # Resize and crop to cell
                img = self._resize_square(img, cell_size - 4)
                x, y = positions[i]
                collage.paste(img, (x + 2, y + 2))

            # Add title
            if title or dog_name:
                draw = ImageDraw.Draw(collage)
                try:
                    font = ImageFont.truetype(self.font_path, 48)
                except:
                    font = ImageFont.load_default()

                text = title or f"{dog_name}'s Week"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                x = (grid_size - text_width) // 2
                y = grid_size - 70

                draw.rectangle([(0, y - 10), (grid_size, grid_size)], fill=(0, 0, 0, 180))
                draw.text((x, y), text, font=font, fill=(255, 255, 255))

            # Save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"collage_{timestamp}.jpg"
            output_path = OUTPUT_DIR / output_filename
            collage.save(output_path, 'JPEG', quality=95)

            return {
                'success': True,
                'output_path': str(output_path),
                'image_count': len(images)
            }

        except Exception as e:
            logger.error(f"Collage creation failed: {e}")
            return {'success': False, 'error': str(e)}


# Singleton
_photo_enhancer = None


def get_photo_enhancer() -> PhotoEnhancer:
    """Get or create PhotoEnhancer instance"""
    global _photo_enhancer
    if _photo_enhancer is None:
        _photo_enhancer = PhotoEnhancer()
    return _photo_enhancer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    enhancer = PhotoEnhancer()

    # Test with a sample image
    test_path = "/home/morgan/dogbot/captures/snapshot_20260110_171926.jpg"
    if os.path.exists(test_path):
        result = enhancer.enhance(
            test_path,
            dog_name="Elsa",
            caption="Such a good girl! Learning to sit on command.",
            filter_name="warm"
        )
        print(f"Result: {result}")
    else:
        print(f"Test image not found: {test_path}")
