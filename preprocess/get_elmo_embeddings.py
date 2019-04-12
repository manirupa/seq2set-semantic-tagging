import argparse
import logging
import os
from typing import List
from tqdm import tqdm

import h5py
import numpy as np
import torch
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.elmo import batch_to_ids
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder

from utils import save

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ElmoTokenEmbedder2(ElmoTokenEmbedder):
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 requires_grad: bool = False,
                 projection_dim: int = None,
                 vocab_to_cache: List[str] = None,
                 scalar_mix_parameters: List[float] = None) -> None:
        super(ElmoTokenEmbedder2, self).__init__(options_file,
                                                 weight_file,
                                                 do_layer_norm,
                                                 dropout,
                                                 requires_grad,
                                                 projection_dim,
                                                 vocab_to_cache,
                                                 scalar_mix_parameters)
        self.mask = None

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, optional.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.

        Returns
        -------
        The ELMo representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)``
        """
        elmo_output = self._elmo(inputs, word_inputs)
        elmo_representations = elmo_output['elmo_representations'][0]
        self.mask = elmo_output['mask']
        if self._projection:
            projection = self._projection
            for _ in range(elmo_representations.dim() - 2):
                projection = TimeDistributed(projection)
            elmo_representations = projection(elmo_representations)
        return elmo_representations


class ElmoEmbedder2:
    def __init__(self,
                 options_file,
                 weight_file,
                 cuda_device,
                 embedding_dim,
                 dropout):

        self.indexer = ELMoTokenCharactersIndexer()
        logger.info("Initializing ELMo.")
        self.elmo = ElmoTokenEmbedder2(options_file, weight_file, dropout=dropout, projection_dim=embedding_dim)
        if cuda_device >= 0:
            self.elmo = self.elmo.cuda(device=cuda_device)

        self.cuda_device = cuda_device
        self.embedding_dim = embedding_dim

    def empty_embedding(self) -> np.ndarray:
        return np.zeros((3, 0, self.embedding_dim))

    def batch_to_embeddings(self, batch):
        character_ids = batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(device=self.cuda_device)

        activations = self.elmo(character_ids)
        mask = self.elmo.mask

        return activations, mask

    def embed_batch(self, batch):
        elmo_embeddings = []

        # Batches with only an empty sentence will throw an exception inside AllenNLP, so we handle this case
        # and return an empty embedding instead.
        if batch == [[]]:
            elmo_embeddings.append(self.empty_embedding())
        else:
            embeddings, mask = self.batch_to_embeddings(batch)
            for i in range(len(batch)):
                length = int(mask[i, :].sum())

                # Slicing the embedding :0 throws an exception so we need to special case for empty sentences.
                if length == 0:
                    elmo_embeddings.append(self.empty_embedding())
                else:
                    elmo_embeddings.append(embeddings[i, :length, :].detach().cpu().numpy())

        return elmo_embeddings

    def embed_sentences(self, sentences, batch_size):
        for batch in lazy_groups_of(iter(sentences), batch_size):
            yield from self.embed_batch(batch)

    def embed_file(self, input_file, output_file_path, output_word_indices, batch_size):
        split_sentences = [line.strip().split() for line in input_file]

        total_words = sum([len(s) for s in split_sentences])

        embedded_sentences = (x for x in self.embed_sentences(split_sentences, batch_size))

        with h5py.File(output_file_path, 'w') as fout:
            dataset = fout.create_dataset(
                'vecs', (total_words, self.embedding_dim), dtype='float32')

            docs = []
            start = 0
            for i, embeddings in enumerate(embedded_sentences):
                sentence_length = embeddings.shape[0]
                end = start+sentence_length
                dataset[start: end] = embeddings
                doc = [i for i in range(start, end)]
                docs.append(doc)
                start = end

            if output_word_indices is not None:
                save(output_word_indices, docs)  # todo: save block by block

    def embed_file_v2(self, input_file, output_file_path, batch_size):
        split_sentences = [line.strip().split() for line in input_file]
        embedded_sentences = ((i, x) for i, x in enumerate(self.embed_sentences(split_sentences, batch_size)))

        with h5py.File(output_file_path, 'w') as fout:
            for i, embeddings in tqdm(embedded_sentences):
                fout.create_dataset(
                    str(i),
                    embeddings.shape, dtype='float32',
                    data=embeddings
                )


def get_hdf5(input_file,
             output_hdf5,
             options_file,
             weight_file,
             embedding_dim,
             dropout,
             batch_size,
             cuda_device):

    ee2 = ElmoEmbedder2(options_file, weight_file, cuda_device, embedding_dim, dropout)

    with open(input_file, encoding='utf-8') as f:
        with torch.no_grad():
            ee2.embed_file_v2(f, output_hdf5, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='elmo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', nargs='?', default='../data/cleaned_no_ids.txt', type=str,
                        help='input file path')
    parser.add_argument('--output_hdf5', nargs='?', default='../data/elmo.hdf5', type=str,
                        help='output hdf5 file path')
    parser.add_argument('--options', nargs='?', default='../data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
                        type=str, help='options json file')
    parser.add_argument('--weights', nargs='?', default='../data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',
                        type=str, help='weights hdf5 file')
    parser.add_argument('--cuda_device', nargs='?', default=-1, type=int, help='gpu id, -1 no gpu')
    parser.add_argument('--batch_size', nargs='?', default=8, type=int, help='batch size')
    parser.add_argument('--embedding_dim', nargs='?', default=100, type=int, help='embedding dim')
    parser.add_argument('--dropout', nargs='?', default=0.3, type=float, help='dropout')
    args = parser.parse_args()

    output_list = [args.output_hdf5]
    for output_file in output_list:
        if output_file is not None and os.path.isfile(output_file):
            raise ValueError("File already exists.{}".format(output_file))

    get_hdf5(args.input,
             args.output_hdf5,
             args.options,
             args.weights,
             args.embedding_dim,
             args.dropout,
             args.batch_size,
             args.cuda_device)
