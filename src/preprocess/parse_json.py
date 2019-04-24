import json
import re


business_id_to_categories = {}

with open('../data/yelp/yelp_academic_dataset_business.json', encoding='utf-8') as f:
    for line in f:
        old_dict = json.loads(line)
        business_id_to_categories[old_dict['business_id']] = old_dict['categories']

with open('../data/yelp/review_and_categories.json', 'w', encoding='utf-8') as f1:
    with open('../data/yelp/yelp_academic_dataset_review.json', encoding='utf-8') as f2:
        for line in f2:
            old_dict = json.loads(line)
            new_dict = {
                # 'business_id': old_dict['business_id'],
                'text': re.sub('[\n ]+', ' ', old_dict['text']),
                'categories': business_id_to_categories[old_dict['business_id']]
            }
            f1.write(json.dumps(new_dict) + '\n')
