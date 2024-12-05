from PIL import Image
import io
import logging

class ImageProcessor:
    @staticmethod
    async def prepare_image(image_data: bytes, max_size: int = 4096, quality: int = 95) -> bytes:
        """
        Prepare image for AI processing by optimizing size and format.
        """
        try:
            image = Image.open(io.BytesIO(image_data))

            # Convert RGBA to RGB if needed
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[3])
                else:
                    background.paste(image, mask=image.split()[1])
                image = background

            # Calculate new dimensions while maintaining aspect ratio
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Save to bytes
            img_byte_arr = io.BytesIO()
            image.save(
                img_byte_arr,
                format='JPEG',
                quality=quality,
                optimize=True,
            )
            
            # Get the byte data
            img_byte_arr.seek(0)
            return img_byte_arr.getvalue()

        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise ValueError(f"Image processing failed: {str(e)}")

    @staticmethod
    def validate_image(image_data: bytes) -> bool:
        """
        Validate if the image data is in a supported format.
        """
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                # Verify the image
                img.verify()
                # Check if format is supported
                if img.format not in ['JPEG', 'PNG', 'WEBP' ,'GIF' ,]:
                    return False
                # Check if size is reasonable
                if img.size[0] * img.size[1] > 25000000:  # Max 25MP
                    return False
                return True
        except Exception as e:
            logging.error(f"Image validation failed: {str(e)}")
            return False

