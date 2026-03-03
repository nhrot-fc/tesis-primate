SPECIES_MAP: dict[str, int] = {
    "aa": 0,
    "ac": 1,
    "as": 2,
    "cc": 3,
    "lw": 4,
    "pt": 5,
    "sb": 6,
    "sm": 7,
}


def get_species_id(specie: str) -> int:
    return SPECIES_MAP[specie] if specie in SPECIES_MAP else -1
