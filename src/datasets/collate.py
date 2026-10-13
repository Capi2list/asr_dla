import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    audio_lengths = torch.Tensor([item["audio"].shape[-1] for item in dataset_items])

    audio_list = [item["audio"].squeeze(0) for item in dataset_items]
    audio = pad_sequence(
        audio_list,
        batch_first=True,
        padding_value=0.0,
    ).unsqueeze(1)

    text_encoded_lengths = torch.tensor(
        [item["text_encoded"].numel() for item in dataset_items], dtype=torch.long
    )

    text_encoded = pad_sequence(
        [item["text_encoded"].reshape(-1) for item in dataset_items],
        batch_first=True,
        padding_value=0,
    )
    spectrogram = pad_sequence(
        [item["spectrogram"].squeeze(0).transpose(0, 1) for item in dataset_items],
        batch_first=True,
        padding_value=0.0,
    ).transpose(1, 2)
    spectrogram_lengths = torch.tensor([[spect.shape[-1] for spect in spectrogram]])

    # TODO remove legacy

    return {
        "raw_audio": [item["raw_audio"] for item in dataset_items],
        "audio": audio,
        "audio_lengths": audio_lengths,
        "spectrogram": spectrogram,
        "spectrogram_length": spectrogram_lengths,
        "text": [item["text"] for item in dataset_items],
        "text_encoded": text_encoded,
        "text_encoded_length": text_encoded_lengths,
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
