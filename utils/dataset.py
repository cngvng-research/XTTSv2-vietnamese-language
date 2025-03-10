import os
import pandas as pd
from datasets import load_dataset
import soundfile as sf
from huggingface_hub import HfApi
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

api = HfApi()

class VietnameseDatasetBuilder:
    def __init__(self, output_dir="vietnamese-datasets", train_split=0.8):
        """
        Khởi tạo class để xây dựng dataset tiếng Việt.

        Args:
            output_dir (str): Thư mục đầu ra để lưu dataset.
            train_split (float): Tỷ lệ dữ liệu dùng cho tập train (phần còn lại là eval).
        """
        self.output_dir = output_dir
        self.wav_dir = os.path.join(output_dir, "wavs")
        self.train_split = train_split
        self.metadata_train = []
        self.metadata_eval = []
        self.speaker_map = {}  # Để gán speaker_id thành dạng @X, @Y, ...
        self.speaker_count = 0
        os.makedirs(self.wav_dir, exist_ok=True)

    def _get_unique_speaker_name(self, speaker_id):
        """
        Gán một speaker_name unique (ví dụ: @X) cho mỗi speaker_id.

        Args:
            speaker_id (str): ID của người nói từ dataset.

        Returns:
            str: Tên speaker unique (e.g., @X).
        """
        if speaker_id not in self.speaker_map:
            self.speaker_map[speaker_id] = f"@{chr(65 + self.speaker_count)}"  # @A, @B, @C, ...
            self.speaker_count += 1
        return self.speaker_map[speaker_id]

    def _convert_to_wav(self, audio_array, sampling_rate, output_path):
        """
        Chuyển đổi dữ liệu âm thanh thành file .wav.

        Args:
            audio_array (np.array): Dữ liệu âm thanh dạng mảng.
            sampling_rate (int): Tần số lấy mẫu.
            output_path (str): Đường dẫn để lưu file .wav.
        """
        sf.write(output_path, audio_array, sampling_rate)

    def process_vietmed(self):
        """
        Xử lý tập dữ liệu VietMed_labeled.
        """
        print("Đang xử lý tập dữ liệu VietMed_labeled...")
        ds = load_dataset("doof-ferb/VietMed_labeled")
        for example in tqdm(ds['train']):
            audio = example['audio']
            transcription = example['transcription']
            speaker_id = example['Speaker ID']
            speaker_name = self._get_unique_speaker_name(speaker_id)
            # Tạo tên file âm thanh unique
            audio_file = f"{speaker_id}_{len(self.metadata_train) + len(self.metadata_eval)}.wav"
            output_path = os.path.join(self.wav_dir, audio_file)
            # Chuyển đổi thành .wav
            self._convert_to_wav(audio['array'], audio['sampling_rate'], output_path)
            # Thêm vào metadata
            self._add_to_metadata(audio_file, transcription, speaker_name)

    def process_vivos(self):
        """
        Xử lý tập dữ liệu vivos.
        """
        print("Đang xử lý tập dữ liệu vivos...")
        ds = load_dataset("AILAB-VNUHCM/vivos")
        for example in tqdm(ds['train']):
            audio = example['audio']
            sentence = example['sentence']
            speaker_id = example['speaker_id']
            speaker_name = self._get_unique_speaker_name(speaker_id)
            # Tạo tên file âm thanh unique
            audio_file = f"{speaker_id}_{len(self.metadata_train) + len(self.metadata_eval)}.wav"
            output_path = os.path.join(self.wav_dir, audio_file)
            # Chuyển đổi thành .wav
            self._convert_to_wav(audio['array'], audio['sampling_rate'], output_path)
            # Thêm vào metadata
            self._add_to_metadata(audio_file, sentence, speaker_name)

    def _add_to_metadata(self, audio_file, text, speaker_name):
        """
        Thêm dữ liệu vào danh sách metadata, chia thành train và eval.

        Args:
            audio_file (str): Tên file âm thanh (e.g., xxx.wav).
            text (str): Câu văn bản tương ứng.
            speaker_name (str): Tên speaker unique (e.g., @X).
        """
        relative_path = os.path.join("wavs", audio_file)
        if len(self.metadata_train) / (len(self.metadata_train) + len(self.metadata_eval) + 1) < self.train_split:
            self.metadata_train.append((relative_path, text, speaker_name))
        else:
            self.metadata_eval.append((relative_path, text, speaker_name))

    def save_metadata(self):
        """
        Lưu metadata vào file CSV.
        """
        print("Đang lưu metadata...")
        train_df = pd.DataFrame(self.metadata_train, columns=["audio_file", "text", "speaker_name"])
        eval_df = pd.DataFrame(self.metadata_eval, columns=["audio_file", "text", "speaker_name"])
        train_df.to_csv(os.path.join(self.output_dir, "metadata_train.csv"), index=False, sep="|")
        eval_df.to_csv(os.path.join(self.output_dir, "metadata_eval.csv"), index=False, sep="|")
        print(f"Metadata đã được lưu vào {self.output_dir}")
        
    def upload_huggingface(self):
        """
        Upload dataset lên Hugging Face.
        """
        print("Đang upload dataset lên Hugging Face...")
        # Check repo exist
        api.create_repo(repo_id="bsmlabs/xttsv2-vietnamese-datasets", private=True, exist_ok=True)
        api.upload_large_folder(
            folder_path=self.output_dir,
            repo_id="bsmlabs/xttsv2-vietnamese-datasets",
            ignore_patterns=["*.py", "*.ipynb"],
            # commit_message="Upload dataset",
            private=True,
            repo_type="dataset"
        )
        print("Dataset đã được upload lên Hugging Face.")
        

    def build(self):
        """
        Thực hiện toàn bộ quá trình xây dựng dataset.
        """
        # self.process_vietmed()
        # # self.process_vivos()
        # self.save_metadata()
        self.upload_huggingface()
        print("Xây dựng dataset hoàn tất.")

# Sử dụng
if __name__ == "__main__":
    builder = VietnameseDatasetBuilder(output_dir="vietnamese-datasets", train_split=0.8)
    builder.build()