"""Vocabulario del dominio: especies, tipos de llamada y sus ids de clase."""

from collections.abc import Iterable
from enum import Enum
from typing import NamedTuple


class Species(Enum):
    AA = "night_monkey"
    AC = "peruvian_spider_monkey"
    AS = "howler_monkey"
    CC = "shock_headed_capuchin_monkey"
    LW = "weddells_saddleback_tamarin"
    PT = "toppins_titi_monkey"
    SB = "bolivian_squirrel_monkey"
    SM = "large_headed_capuchin"


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
}


class CallTypeNote(NamedTuple):
    meaning: str
    selection: str


# Notas de campo del protocolo de anotación: qué significa cada llamada y qué criterio
# usaron los anotadores para dibujar la caja. No cubre todos los códigos de `CALL_TYPES`,
# solo los que tienen nota documentada.
CALL_TYPE_NOTES: dict[tuple[Species, str], CallTypeNote] = {
    (Species.AC, "chc"): CallTypeNote(
        meaning="Warning call to acknowledge terrestrial danger (researchers), not an alarm call.",
        selection="Entire sequence, varying length.",
    ),
    (Species.AC, "gc"): CallTypeNote(
        meaning="Interaction between two individuals associated with dominance/play. "
        "Mostly noticed in juvenile plays.",
        selection="Entire sequence with harmonics, different length.",
    ),
    (Species.AC, "cc"): CallTypeNote(
        meaning="Done to maintain cohesion in the group.",
        selection="Spaced call, with harmonics.",
    ),
    (Species.AC, "whc"): CallTypeNote(
        meaning="Long distance vocalization. Emitted in different circumstances: group "
        "movements, feeding, or contact maintenance.",
        selection="Entire sequence with harmonics, different length.",
    ),
    (Species.AC, "bc"): CallTypeNote(
        meaning="Passive: also a contact call. Aggressive: long and loud, a warning call, "
        "possible predator close by, or when juveniles lost the troop.",
        selection="Spaced calls with harmonics.",
    ),
    (Species.LW, "cc"): CallTypeNote(
        meaning="Done to maintain cohesion in the group; often followed by a response; "
        "3-7 syllables.",
        selection="Entire sequence, different lengths.",
    ),
    (Species.LW, "cs"): CallTypeNote(
        meaning="Individual syllable of a contact_call (cc) sequence.",
        selection="Individual syllable, nested inside its contact_call sequence.",
    ),
    (Species.LW, "aa"): CallTypeNote(
        meaning="Done to acknowledge aerial danger, no response.",
        selection="Short sound; back tail longer than front.",
    ),
    (Species.LW, "ta"): CallTypeNote(
        meaning="Done to acknowledge terrestrial danger, no response.",
        selection="Entire sequence; oscillating wave appearance.",
    ),
    (Species.LW, "sqc"): CallTypeNote(
        meaning="Squeals seem to reduce the hostility of conspecifics.",
        selection="Isolated call, with harmonics.",
    ),
    (Species.LW, "phc"): CallTypeNote(
        meaning="Vocalizations emitted during loose-contact interactions, likely from an "
        "immature individual.",
        selection="Entire sequence.",
    ),
    (Species.LW, "tr"): CallTypeNote(
        meaning="Contact calls over small distances.",
        selection="Entire sequence, first syllable bigger than the rest, 2-4 syllables, "
        "variations.",
    ),
    (Species.LW, "tf"): CallTypeNote(
        meaning="Contact calls over small distances.",
        selection="Entire sequence, last syllable length longer than the rest, variations.",
    ),
    (Species.LW, "tt"): CallTypeNote(
        meaning="Contact calls over small distances.",
        selection="Entire sequence, first syllable length bigger than the rest, variations.",
    ),
    (Species.LW, "vc"): CallTypeNote(
        meaning="Visual acknowledgment, occurs during undisturbed social contexts.",
        selection="Entire sequence.",
    ),
    (Species.PT, "dc"): CallTypeNote(
        meaning="Territorial call made in chorus by a male and female couple.",
        selection="Entire sequence, mix of bellow phrases (straight lines - female) and "
        "pant phrases (tower - male). Select the harmonic too.",
    ),
    (Species.PT, "ac"): CallTypeNote(
        meaning="Functions as a deterrent signal to predators. Could also inform other "
        "individuals about location and distance of the predator.",
        selection="Entire sequence, with harmonics.",
    ),
    (Species.AS, "hc"): CallTypeNote(
        meaning="Long territorial call.",
        selection="Entire sequence.",
    ),
    (Species.AS, "bc"): CallTypeNote(
        meaning="Warning call, sometimes happens in the introduction of the howl call.",
        selection="Spaced calls.",
    ),
    (Species.SB, "pcc"): CallTypeNote(
        meaning="Done to maintain cohesion in the group.",
        selection="Entire sequence. Can take different shapes. Different length.",
    ),
    (Species.SB, "ppc"): CallTypeNote(
        meaning="Associated with movement and play.",
        selection="Spaced calls. Can take different shapes.",
    ),
    (Species.SB, "lpc"): CallTypeNote(
        meaning="Warning call, when they lose visual contact.",
        selection="Isolated calls, entire sequence.",
    ),
    (Species.SB, "spc"): CallTypeNote(
        meaning="The most used affiliative call.",
        selection="Spaced calls, very high pitch sounds.",
    ),
    (Species.SB, "sc"): CallTypeNote(
        meaning="Extreme distress or fear. Individuals beaten in a fight or restrained. "
        "During or before an attack.",
        selection="Isolated calls, very high pitch sounds.",
    ),
    (Species.SM, "cc"): CallTypeNote(
        meaning="Done to maintain cohesion in the group.",
        selection="Spaced calls with harmonics.",
    ),
    (Species.SM, "acc"): CallTypeNote(
        meaning="Produced during mild aggression directed at a close member. Could involve "
        "branch shaking, fixed stare, open mouth bared teeth. Alpha male and female adults "
        "towards juveniles and infants.",
        selection="Entire sequence without harmonics.",
    ),
    (Species.SM, "sc"): CallTypeNote(
        meaning="Given in response to aggression.",
        selection="High variability in duration and frequency. Entire sequence, long "
        "series, select harmonics.",
    ),
    (Species.SM, "pc"): CallTypeNote(
        meaning="Emitted when the troop is feeding on clumped resources.",
        selection="Entire sequence.",
    ),
    (Species.SM, "whc"): CallTypeNote(
        meaning="Contact call.",
        selection="Entire sequence. Different length.",
    ),
    (Species.SM, "hic"): CallTypeNote(
        meaning="Warning call to acknowledge terrestrial danger.",
        selection="Entire sequence. Spaced calls. Sound wave descending from left to right.",
    ),
    (Species.SM, "fc"): CallTypeNote(
        meaning="Foraging, food found in high concentration.",
        selection="Entire sequence. Different lengths. Consider harmonics if clear enough. "
        "Careful with Trogon or Fasciated Antshrike sounds.",
    ),
    (Species.SM, "fs"): CallTypeNote(
        meaning="Individual syllable of a food_call (fc) sequence.",
        selection="Individual syllable, nested inside its food_call sequence.",
    ),
    (Species.AA, "gc"): CallTypeNote(
        meaning="Done to maintain the spatial cohesion of the group. Female.",
        selection="Spaced calls. Low frequency. Consider the bottom point.",
    ),
    (Species.AA, "hm"): CallTypeNote(
        meaning="Done to maintain the spatial cohesion of the group. Male.",
        selection="Spaced calls. Sounds like a heartbeat.",
    ),
    (Species.AA, "sc"): CallTypeNote(
        meaning="Warning call to acknowledge terrestrial danger.",
        selection="Entire sequence. Looks like 2 or 3 vertical lines. Sounds like an insect.",
    ),
    (Species.CC, "cc"): CallTypeNote(
        meaning="Done to maintain cohesion in the group.",
        selection="Spaced calls with harmonics.",
    ),
}

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
