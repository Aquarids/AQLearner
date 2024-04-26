from torch.utils.data import Dataset

class WikiText2(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return (len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.seq_length
        sequence = self.data[start_idx:end_idx + 1]
        input_seq = sequence[:-1]  # All but the last token
        target_seq = sequence[1:]  # All but the first token
        return input_seq, target_seq

