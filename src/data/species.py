from collections.abc import Iterable
from enum import Enum


class Species(Enum):
    AA = "night_monkey"  # Aotus azarae
    AC = "peruvian_spider_monkey"  # Ateles chamek
    AS = "howler_monkey"  # Alouatta sara
    CC = "shock_headed_capuchin_monkey"  # Cebus cuscinus
    LW = "weddells_saddleback_tamarin"  # Leontocebus weddelli
    PT = "toppins_titi_monkey"  # Plecturocebus toppini
    SB = "bolivian_squirrel_monkey"  # Saimiri boliviensis peruviensis
    SM = "large_headed_capuchin"  # Sapajus macrocephalus
    AV = "bird"  # PteroSet (Ruiz et al. 2026): aves neotropicales, sólo como fondo


# Tipos de llamada válidos por especie: código en la anotación -> nombre legible.
CALL_TYPES: dict[Species, dict[str, str]] = {
    Species.AA: {
        "gc": "gulp_call",
        "hm": "hoot_call",
        "sc": "squeak_call",
    },
    Species.AC: {
        "chc": "chitter_call",
        "gc": "growl_call",
        "cc": "contact_call",
        "whc": "whinnie_call",
        "sc": "squeak_call",
        "bc": "bark_call",
    },
    Species.AS: {"hc": "howl_call", "bc": "bark_call"},
    Species.CC: {"cc": "contact_call"},
    Species.LW: {
        "cc": "contact_call",
        "cs": "contact_syllable",
        "aa": "aerial_alarm_call",
        "ta": "terrestrial_alarm_call",
        "sqc": "squeal_call",
        "phc": "phee_call",
        "tr": "trino_call",
        "tf": "trino_fast_call",
        "tj": "unofficial_tj_call",
        "tt": "trino_transition_call",
        "vc": "visual_contact_call",
    },
    Species.PT: {
        "dc": "duet_call",
        "ac": "alarm_call",
        "bp": "unofficial_bellow_phrase",
        "pp": "unofficial_pant_phrase",
        "sqc": "unofficial_squeal_call",
    },
    Species.SB: {
        "pcc": "peep_contact_call",
        "ppc": "play_peep_call",
        "lpc": "long_peep_call",
        "spc": "spit_call",
        "sc": "shriek_call",
    },
    Species.SM: {
        "cc": "contact_call",
        "acc": "aggressive_contact_call",
        "sc": "squeal_call",
        "pc": "purr_call",
        "whc": "whistle_call",
        "hic": "hip_call",
        "fc": "food_call",
        "fs": "food_syllable",
    },
    Species.AV: {"voc": "bird_vocalization"},
}

# Especies que no son clase: sus cajas marcan qué ventanas tienen un sonido que el detector
# tiene que aprender a ignorar, y entran al caché como ventanas sin cajas.
BACKGROUND_SPECIES: frozenset[str] = frozenset({Species.AV.name.lower()})


# Pares (especie, tipo) admitidos, en minúsculas como vienen del `.txt` limpio.
VALID_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (species.name.lower(), code) for species, codes in CALL_TYPES.items() for code in codes
)


class LabelSet:
    def __init__(self, names: Iterable[str]):
        self.names = sorted({str(name) for name in names})
        self._ids = {name: index for index, name in enumerate(self.names)}

    def id(self, name: str) -> int:
        return self._ids[str(name)]

    def name(self, class_id: int) -> str:
        return self.names[int(class_id)]

    def __contains__(self, name: object) -> bool:
        return str(name) in self._ids

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return f"LabelSet({len(self)} clases: {', '.join(self.names)})"
