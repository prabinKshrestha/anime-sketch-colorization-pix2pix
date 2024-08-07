from train import train_pix2pix
from test import test_and_save_outputs_using_all_models, test_and_save_outputs_using_last_model
from eval import execute_evaluation_for_last_model, execute_evaluation_for_all_models

#####################################################################################

# MAIN Configs

TEST_FOR_EACH_SAVED_MODEL = True

#####################################################################################

def set_test_epoch_frequency(value = True):
    global TEST_FOR_EACH_SAVED_MODEL
    TEST_FOR_EACH_SAVED_MODEL = value

def start():
    print("##########################################################################################")
    print("##########################################################################################")
    print("##########################################################################################")
    print("######### Starting Anime Sketch Colorization Pix2Pix #########")
    print("##########################################################################################")

    print("########### Start Training pix2pix ###########")
    train_pix2pix()
    print("########### Complete Training pix2pix ###########")

    print("##########################################################################################")

    if TEST_FOR_EACH_SAVED_MODEL:
        print("########### Start Estimation and Save output images for all saved models ###########")
        test_and_save_outputs_using_all_models()
    else:
        print("########### Start Estimation and Save output images for final saved model ###########")
        test_and_save_outputs_using_last_model
    print("########### Complete Estimation ###########")

    print("##########################################################################################")

    if TEST_FOR_EACH_SAVED_MODEL:
        print("########### Start Evaluation for all saved models ###########")
        execute_evaluation_for_all_models()
    else:
        print("########### Start Evaluation for final saved model ###########")
        execute_evaluation_for_last_model()
    print("########### Complete Evaluation ###########")

    print("##########################################################################################")
    print("######### Completing Anime Sketch Colorization Pix2Pix #########")
    print("##########################################################################################")
    print("##########################################################################################")
    print("##########################################################################################")


if __name__ == "__main__":
    start()
