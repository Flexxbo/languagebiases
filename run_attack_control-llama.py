from utils import read_products, products_to_string, align, get_tokenizer_aws_model_id, generate_response
from control_test import SYSTEM_PROMPT, create_prompt, get_user_msg
import os 
import random
import pickle
from tqdm import tqdm 
import argparse
import pandas as pd


from boosting_prompts import social_proof, exclusivity, attack_scarcity, attack_discount_framing, contrast_effect, decoy_effect, authority_bias, bizarreness_effect, denominator_neglect, storytelling_effect, identity_signaling

from attacks_baseline import STS, social_proof_baseline, exclusivity_baseline, attack_scarcity_baseline, attack_discount_framing_baseline, bizarreness_effect_baseline, contrast_effect_baseline, decoy_effect_baseline,  identity_signaling_baseline, authority_bias_baseline, control_baseline


import sys 
import os
import time

base_path = "./outputs_rank_optimizer"


def attack_is_complete(model, attack_type, query_type, product):
    
    for attacked_prd in range (10):
        try:
            with open(f"outputs_rank_optimizer/{attack_type}/experiment_{product}_{query_type}_{model}_{attack_type}_{attacked_prd}.pickle", "rb") as handle:
                da = pickle.load(handle)
        except:
            return False
        
    return True 




def attack_experiment(model_name, attack, aws_keys_csv_filename, run_control): ###def attack_experiment(model_name, attack, catalog_type, aws_keys_csv_filename, run_control):
    
    
    aws_keys_csv = pd.read_csv(aws_keys_csv_filename)
    aws_access_key_id = aws_keys_csv["Access key ID"].iloc[0]
    aws_secret_access_key = aws_keys_csv["Secret access key"].iloc[0]
    
    attacks_mapping = {
        "social_proof": social_proof,
        "exclusivity": exclusivity, 
        "attack_scarcity": attack_scarcity, 
        "attack_discount_framing": attack_discount_framing, 
        "contrast_effect": contrast_effect, 
        "decoy_effect": decoy_effect, 
        "authority_bias": authority_bias,
        "bizarreness_effect": bizarreness_effect,
        "denominator_neglect": denominator_neglect,
        "storytelling_effect": storytelling_effect,
        "identity_signaling": identity_signaling,
        
        "social_proof_baseline": social_proof_baseline,
        "exclusivity_baseline": exclusivity_baseline,
        "attack_scarcity_baseline": attack_scarcity_baseline,
        "attack_discount_framing_baseline": attack_discount_framing_baseline,
        "bizarreness_effect_baseline": bizarreness_effect_baseline,
        "contrast_effect_baseline": contrast_effect_baseline,
        "decoy_effect_baseline": decoy_effect_baseline,
        "authority_bias_baseline": authority_bias_baseline,
        "identity_signaling_baseline": identity_signaling_baseline,
        "control_attack_baseline": control_baseline,
        
    }
    
    if run_control == False:
        if attack.replace("_defence", "") in attacks_mapping:
            attacker = attacks_mapping[attack.replace("_defence", "") ]
        else:
            raise(Exception(f"Atack is not known! Choose one of the following: {', '.join(attacks_mapping.keys())}"))
   
    
    output_folder = os.path.join(base_path, attack)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
                        
    tokenizer, aws_model_id = get_tokenizer_aws_model_id(model_name)

    for catalog in ['coffee_machines']: ###catalog_ = ['coffee_machines', 'cameras', 'books'] if catalog_type == 'simple' else ["laptops", "home_office_chairs", "chew_toys"] for catalog in catalog_:
        for user_msg_type in ["abstract"]:
            if attack_is_complete(model_name, attack, user_msg_type, catalog):
                print (f"Attack is complete: {[model_name, attack, user_msg_type, catalog]}")
                continue

            for item_attacked in range (10): ###user_msg, filename = get_user_msg(catalog, user_msg_type) #list_of_products = read_products(filename) #for item_attacked in range (len(list_of_products)):
                outps = []
                for i in range (5):
                    print(f"Running catalog={catalog}, attack={attack}, item={item_attacked}, run={i+1}", flush=True)
                    user_msg, filename = get_user_msg(catalog, user_msg_type) ###delete from here
                    list_of_products = read_products(filename) # read products ###delete from here
                        
                    if "baseline" in attack:
                        if run_control == False:
                            list_of_products[item_attacked]["Description"] = attacker(list_of_products[item_attacked], aws_access_key_id, aws_secret_access_key) # attack to the product
                            #time.sleep(15)
                        list_of_products_shuffled = list_of_products.copy() # make a copy to shuffle it
                    else:
                        list_of_products_exp = read_products(filename.split(".json")[0] + "_expanded.jsonl") # read products ### :( 
                        if run_control == False:    
                            list_of_products_exp[item_attacked]["Description"] = attacker(list_of_products[item_attacked], aws_access_key_id, aws_secret_access_key) # attack to the product
                            time.sleep(10)
                        list_of_products_shuffled = list_of_products_exp.copy() # make a copy to shuffle it
                        
                    random.shuffle(list_of_products_shuffled) # shuffle the list of products 

                    products = create_prompt(list_of_products_shuffled, user_msg)
                    llm_response = generate_response(products, system_prompt = SYSTEM_PROMPT, tokenizer = tokenizer, aws_model_id = aws_model_id, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

                    al = align(llm_response, list_of_products)  ### need an if for fuzzy_align
                    outps.append([
                        filename, list_of_products_shuffled, products, llm_response, al
                    ])

                    with open (os.path.join(output_folder, f"experiment_{catalog}_{user_msg_type}_{model_name}_{attack}_{item_attacked}.pickle"), "wb") as handle:
                        pickle.dump(outps, handle)
                    time.sleep(10)
                if run_control == True:
                    break
                            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Code for implementing an attack.")
    parser.add_argument('--model_name', type=str, required=True, help="The name of the model. Supported models:['llama-8b', 'llama-70b', 'llama-405b']")
    parser.add_argument('--attack', type=str, required=True, help="The name of the attack.")
    parser.add_argument('--aws_keys_csv_filename', type=str, required=True, help="The path for the .csv that has the aws keys.")
    parser.add_argument('--run_control', type=bool, default=False, help="Declares if you want to run a control experiment")
                        
    args = parser.parse_args()
    model_name = args.model_name
    attack = args.attack
    aws_keys_csv_filename = args.aws_keys_csv_filename    
    run_control = args.run_control
    # run the experiment 
    attack_experiment(model_name, attack, aws_keys_csv_filename, run_control)
    
    
    
    
