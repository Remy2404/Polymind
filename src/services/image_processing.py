from PIL import Image
import io

class ImageProcessor:
    @staticmethod
    def prepare_image(image_data: bytes) -> bytes:
        image = Image.open(io.BytesIO(image_data))
        
        # Resize if image is too large
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        return img_byte_arr.getvalue()
