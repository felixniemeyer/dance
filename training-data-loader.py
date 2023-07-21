import torchaudio

from torch.utils.data import Dataset, DataLoader

class CustomAudioDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return len(self.audio_file_paths)

    def __getitem__(self, index):
        # Load and preprocess audio file
        audio, sample_rate = torchaudio.load(self.audio_file_paths[index])
        # Apply audio preprocessing (e.g., convert to spectrogram, normalize, etc.)
        if self.transform is not None:
            audio = self.transform(audio)

        # Load and preprocess label information
        with open(self.label_file_paths[index], 'r') as file:
            text_info = file.read()
        # Tokenize text and convert to class labels
        class_labels = self.tokenizer.tokenize(text_info)

        return audio, class_labels

# Example usage:
# train_dataset = CustomAudioDataset(audio_file_paths_train, label_file_paths_train, tokenizer, transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
