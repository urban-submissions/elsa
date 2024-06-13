import json
import numpy as np
"""
result
array(['alone', 'at petrol/gas station', 'baby/infant', 'biking',
       'construction workers', 'couple/2people', 'crossing crosswalk',
       'dining', 'duplicate', 'elderly', 'group', 'gsv car interaction',
       'kid', 'mobility aids', 'model_hint', 'multi-label',
       'no interaction', 'no people', 'not sure/confusing',
       'on wheelchair', 'pet', 'pet interactions', 'phone interaction',
       'playing', 'public service/cleaning',
       'pushing stroller or shopping cart', 'pushing wheelchair',
       'pushing wheelchair or stroller', 'riding carriage', 'running',
       'shopping', 'sitting', 'sport activities', 'standing',
       'street vendors', 'taking photo', 'talking', 'talking on phone',
       'teenager', 'waiting in bus station', 'walking', 'with bike',
       'with coffee or drinks', 'working/laptop'], dtype=object)
"""
condition = 'condition'
state = 'state'
activity = 'activity'
others = 'others'

label2meta = {
    'alone': condition,
    'at petrol/gas station': activity,
    'baby/infant': others,
    'biking': state,
    'construction workers': others,
    'couple/2people': condition,
    'crossing crosswalk': activity,
    'dining': activity,
    'duplicate': others,
    'elderly': others,
    'group': condition,
    'gsv car interaction': activity,
    'kid': others,
    'mobility aids': state,
    'model_hint': others,
    'multi-label': others,
    'no interaction': activity,
    'no people': condition,
    'not sure/confusing': others,
    'on wheelchair': state,
    'pet': others,
    'pet interactions': activity,
    'phone interaction': activity,
    'playing': activity,
    'public service/cleaning': activity,
    'pushing stroller or shopping cart': activity,
    'pushing wheelchair': activity,
    'pushing wheelchair or stroller': activity,
    'riding carriage': state,
    'running': state,
    'shopping': activity,
    'sitting': state,
    'sport activities': activity,
    'standing': state,
    'street vendors': others,
    'taking photo': activity,
    'talking': activity,
    'talking on phone': activity,
    'teenager': others,
    'waiting in bus station': activity,
    'walking': state,
    'with bike': activity,
    'with coffee or drinks': activity,
    'working/laptop': activity,
    #### additional classes from the original file
    'picnic': activity,
    'police': activity,
    'reading': activity,
    'snacking': activity,
    'taking cab/taxi': activity,
    'waiting for food/drinks': activity,
    'with luggage': others,
    'load/unload packages from car/truck': activity,
    'hugging': activity,
    'riding motorcycle': state
}

meta2imeta = {
    'condition': 0,
    'state': 1,
    'activity': 2,
    'others': 3,
}

wrong2ambiguities = {
    "wheelchair": "on wheelchair",
    "vendor": "street vendors",
    "teenager": "teenager",
    "stroller": [
        "pushing stroller or shopping cart",
        "pushing wheelchair or stroller"
    ],
    "street vendor": "street vendors",
    "street": "no people",
    "sport activities": "sport activities",
    "public service": "public service/cleaning",
    "photo": "taking photo",
    "phone interaction": "phone interaction",
    "phone": "talking on phone",
    "pet interactions": "pet interactions",
    "pet": "pet",
    "person": "alone",
    "people": "group",
    "old": "elderly",
    "group": "group",
    "gas station": "at petrol/gas station",
    "crosswalk": "crossing crosswalk",
    "couple": "couple/2people",
    "construction workers": "construction workers",
    "child": "kid",
    "cart": "pushing stroller or shopping cart",
    "carriage": "riding carriage",
    "bus station": "waiting in bus station",
    "bike": [
        "biking",
        "with bike"
    ],
    "baby": "baby/infant"
}

# reverse alphabetical so that larger strings get priority
wrong2ambiguities = dict(sorted(wrong2ambiguities.items(), key=lambda item: item[0], reverse=True))
obj = json.dumps(wrong2ambiguities, indent=4)
wrong2ambiguities = {
    key: value
    if isinstance(value, list)
    else [value]
    for key, value in wrong2ambiguities.items()
}
