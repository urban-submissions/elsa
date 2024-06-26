def syn(synonyms: list[str]) -> list[set[str]]:
    return [
        set(syn.split("; "))
        for syn in synonyms
    ]


PROMPTS = syn([
    'individual; person',
    'at petrol station; at gas station',
    'biking; bicycling',
    'construction-worker',
    'two people',
    'dining; eating',
    'car interaction',
    'group; people; gathering',
    'on wheelchair; riding wheelchair',
    'with pet; with dog',
    "looking at phone",
    'playing',
    "pushing stroller",
    'running; jogging',
    'shopping',
    'sitting on chair',
    'sports',
    'standing',
    'street-vendor; street-merchant',
    'taking picture; photographing',
    'talking; chatting',
    'talking on phone; chatting on phone',
    'waiting for the bus; waiting at bus station',
    'walking',
    'with bike; with bicycle',
    'with drink; with beverage',
    'baby; infant',
    # 'crossing street; crossing crosswalk',
    'crossing crosswalk',
    'duplicate; duplicated',
    'elderly; senior',
    'kid; child',
    'with cane or walker',
    'model_hint',
    'multi-label',
    'no people',
    'not sure; confusing; ambiguous',
    'pet; service dog',
    'with pet; with dog'
    'public service; cleaning',
    'riding carriage; riding horse carriage',
    'teenager; teen',
    'working on laptop; working on computer',
    'no interaction',
    'police; law enforcement',
    'loading packages; unloading packages',
    'reading; reading book',
    'with luggage; with suitcase',
    'waiting for food; waiting for drink',
    'taking taxi; taking cab',
    'picnic; picnicking',
    'riding motorcycle; motorcycling',
    'hugging; embracing',

])
