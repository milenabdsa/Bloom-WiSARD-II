from typing import List, Tuple
import numpy as np

from discriminator import Discriminator

class Model:
    def __init__(self, input_params):
        self.num_pc_filters = 3
        self.num_lhr_filters = 3
        self.num_ghr_filters = 3
        self.num_ga_filters = 3
        self.num_xor_filters = 3
        self.pc_lut_addr_size = input_params[0]
        self.lhr_lut_addr_size = input_params[1]
        self.ghr_lut_addr_size = input_params[2]
        self.ga_lut_addr_size = input_params[3]
        self.xor_lut_addr_size = input_params[4]
        self.pc_num_hashes = 3
        self.lhr_num_hashes = 3
        self.ghr_num_hashes = 3
        self.ga_num_hashes = 3
        self.xor_num_hashes = 3
        self.ghr_size = input_params[5]
        self.ga_branches = input_params[6]
        self.seed = 203

        self.pc_discriminators = [
            Discriminator(self.num_pc_filters, self.pc_lut_addr_size, self.pc_num_hashes)
            for _ in range(2)
        ]

        self.xor_discriminators = [
            Discriminator(self.num_xor_filters, self.xor_lut_addr_size, self.xor_num_hashes)
            for _ in range(2)
        ]

        self.lhr_discriminators = [
            Discriminator(self.num_lhr_filters, self.lhr_lut_addr_size, self.lhr_num_hashes)
            for _ in range(2)
        ]

        self.ghr_discriminators = [
            Discriminator(self.num_ghr_filters, self.ghr_lut_addr_size, self.ghr_num_hashes)
            for _ in range(2)
        ]

        self.ga_discriminators = [
            Discriminator(self.num_ga_filters, self.ga_lut_addr_size, self.ga_num_hashes)
            for _ in range(2)
        ]

        #self.ghr_size = 24
        self.ghr = np.zeros(self.ghr_size, dtype=np.uint8)

        self.lhr_configs = [
            (24, 12),  # (comprimento, bits_pc) para LHR1
            (9, 9),  # LHR2
            (5, 5),  # LHR3
        ]

        self.lhrs = []
        for length, bits_pc in self.lhr_configs:
            lhr_size = 1 << bits_pc
            self.lhrs.append(np.zeros((lhr_size, length), dtype=np.uint8))

        self.ga_lower = 8
        self.ga = np.zeros(self.ga_lower * self.ga_branches, dtype=np.uint8)

        self.input_size = (
            24
            + self.ghr_size
            + 24
            + sum(self.lhr_configs[i][0] * input_params[i + 2] for i in range(3))
            + len(self.ga)
        )

    def extract_features(self, pc: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        pc_bits = np.array(
            [int(b) for b in format(pc & ((1 << 24) - 1), "024b")], dtype=np.uint8
        )
        #pc_bits_repeated = np.tile(pc_bits, self.pc_times)

        # LHRs
        lhr_features = []
        #lhr_times_list = [
        #    self.lhr1_times,
        #    self.lhr2_times,
        #    self.lhr3_times,
        #]
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            #if lhr_times_list[i] > 0:
                index = int("".join(map(str, pc_bits[-bits_pc:])), 2)
                lhr = self.lhrs[i][index]
                #lhr_repeated = np.tile(lhr, lhr_times_list[i])
                lhr_features.append(lhr)
        lhr_features_combined = (
            np.concatenate(lhr_features) if lhr_features else np.array([], dtype=np.uint8)
        )

        #ghr_repeated = np.tile(self.ghr, self.ghr_times)
        effective_xor_len = min(self.ghr_size, len(pc_bits))
        pc_bits_for_xor = pc_bits[-effective_xor_len:]
        ghr_for_xor = self.ghr[-effective_xor_len:]
        pc_ghr_xor = np.bitwise_xor(pc_bits_for_xor, ghr_for_xor)
        #pc_ghr_xor_repeated = np.tile(pc_ghr_xor, self.pc_ghr_times)
        
        #ga_repeated = (
        #    np.tile(self.ga, self.ga_times) if self.ga_times > 0 else np.array([], dtype=np.uint8)
        #)
        #ghr_ga_features = np.concatenate([self.ghr, self.ga])

        return pc_bits, pc_ghr_xor, lhr_features_combined, self.ghr, self.ga

    def get_input_pieces(
        self,
        pc_features: np.ndarray,
        xor_features: np.ndarray,
        lhr_features: np.ndarray,
        ga_features: np.ndarray,
        ghr_features: np.ndarray,
    ) -> Tuple[List[bytes], List[bytes], List[bytes]]:  # Retorna tupla de listas
        pc_pieces = self._get_pieces(pc_features, self.num_pc_filters, self.seed)
        lhr_pieces = self._get_pieces(lhr_features, self.num_lhr_filters, self.seed)
        xor_pieces = self._get_pieces(xor_features, self.num_xor_filters, self.seed)
        ghr_pieces = self._get_pieces(ghr_features, self.num_ghr_filters, self.seed) 
        ga_pieces = self._get_pieces(ga_features, self.num_ga_filters, self.seed) 
        return pc_pieces, xor_pieces, lhr_pieces, ghr_pieces, ga_pieces

    def _get_pieces(self, features: np.ndarray, num_filters: int, seed: int) -> List[bytes]:
        binary_input = "".join(list(map(str, features.tolist())))
        #indices = list(range(len(binary_input)))
        #random.seed(seed)
        #random.shuffle(indices)
        #shuffled_binary = "".join(binary_input[i] for i in indices)
        chunk_size = len(binary_input) // num_filters
        chunks = [
            binary_input[i * chunk_size : (i + 1) * chunk_size]
            for i in range(num_filters)
        ]
        remainder = len(binary_input) % num_filters
        for i in range(remainder):
            chunks[i] += binary_input[num_filters * chunk_size + i]
        return [chunk.encode() for chunk in chunks]

    def predict_and_train(self, pc: int, outcome: int, hierarchy: List[str]) -> bool:
        
        pc_features, xor_features, lhr_features, ghr_features, ga_features = self.extract_features(pc)
        pc_pieces, xor_pieces, lhr_pieces, ghr_pieces, ga_pieces = self.get_input_pieces(
            pc_features, xor_features, lhr_features, ga_features, ghr_features
        )

        
        pc_count_0=self.pc_discriminators[0].get_count(pc_pieces)
        pc_count_1=self.pc_discriminators[1].get_count(pc_pieces)
        lhr_count_0= self.lhr_discriminators[0].get_count(lhr_pieces)
        lhr_count_1= self.lhr_discriminators[1].get_count(lhr_pieces)
        ghr_count_0= self.ghr_discriminators[0].get_count(ghr_pieces)
        ghr_count_1= self.ghr_discriminators[1].get_count(ghr_pieces)
        ga_count_0=self.ga_discriminators[0].get_count(ga_pieces)
        ga_count_1=self.ga_discriminators[1].get_count(ga_pieces)
        xor_count_0= self.xor_discriminators[0].get_count(xor_pieces)
        xor_count_1= self.xor_discriminators[1].get_count(xor_pieces)
        

        predictions = {
            'pc': 1 if pc_count_1 > pc_count_0 else 0,
            'lhr': 1 if lhr_count_1 > lhr_count_0 else 0,
            'ghr': 1 if ghr_count_1 > ghr_count_0 else 0,
            'ga': 1 if ga_count_1 > ga_count_0 else 0,
            'xor': 1 if xor_count_1 > xor_count_0 else 0
        }

        # Lógica de fusão de 3 passos
    
        # Passo 1: Comparar os dois primeiros preditores
        if predictions[hierarchy[0]] == predictions[hierarchy[1]]:
            prediction = predictions[hierarchy[0]]
        
        # Passo 2: Se não houver consenso entre os dois primeiros,
        # verifique se os próximos dois concordam com a decisão de um deles.
        elif (predictions[hierarchy[2]] == predictions[hierarchy[0]] and 
            predictions[hierarchy[3]] == predictions[hierarchy[0]]):
            prediction = predictions[hierarchy[0]]
            
        elif (predictions[hierarchy[2]] == predictions[hierarchy[1]] and
            predictions[hierarchy[3]] == predictions[hierarchy[1]]):
            prediction = predictions[hierarchy[1]]
        
        # Passo 3: Se não houver consenso nos passos anteriores,
        # a decisão final é do último preditor na hierarquia.
        else:
            prediction = predictions[hierarchy[4]]

        if prediction != outcome:
            self.pc_discriminators[outcome].train(pc_pieces)
            self.lhr_discriminators[outcome].train(lhr_pieces)
            self.ghr_discriminators[outcome].train(ghr_pieces)
            self.ga_discriminators[outcome].train(ga_pieces)
            self.xor_discriminators[outcome].train(xor_pieces)

            self.pc_discriminators[prediction].forget(pc_pieces)
            self.lhr_discriminators[prediction].forget(lhr_pieces)
            self.ghr_discriminators[prediction].forget(ghr_pieces)
            self.ga_discriminators[prediction].forget(ga_pieces)
            self.xor_discriminators[prediction].forget(xor_pieces)

        self._update_histories(pc, outcome)
        return prediction == outcome

    def apply_bleaching(self):
        for disc in self.pc_discriminators:
            disc.binarize(self.pc_bleaching_threshold)

        for disc in self.lhr_discriminators:
            disc.binarize(self.lhr_bleaching_threshold)

        for disc in self.ghr_discriminators:
            disc.binarize(self.ghr_bleaching_threshold)

        for disc in self.ga_discriminators:
            disc.binarize(self.ga_bleaching_threshold)

        for disc in self.xor_discriminators:
            disc.binarize(self.xor_bleaching_threshold)

    def _update_histories(self, pc: int, outcome: int):
        self.ghr = np.roll(self.ghr, -1)
        self.ghr[-1] = outcome

        pc_bits = np.array(
            [int(b) for b in format(pc & ((1 << 24) - 1), "024b")], dtype=np.uint8
        )
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int("".join(map(str, pc_bits[-bits_pc:])), 2)
            self.lhrs[i][index] = np.roll(self.lhrs[i][index], -1)
            self.lhrs[i][index][-1] = outcome

        new_bits = pc_bits[-self.ga_lower :]
        self.ga = np.roll(self.ga, -self.ga_lower)
        self.ga[-self.ga_lower :] = new_bits