Python code for running offline BMI experiments with the Chestek lab BMI rig data. Collection of functionalities originally from pybmi.

# Possible things to change/discuss in the structure design

- decide if the train_model should be defined inside the model definition, as I thought about it, or if we want to move it to a static method inside a TrainingUtilities class. The benefit of definition inside each model is that we can specify the training modalities specific for each model, and the training of the model will be independet of these: you just call model.train_model. The "problem" of this is possible repetitions of training strategies for each model. A solution to this last problem could be to define a collection of common steps for training in the NeuralNetwork abstract class, and use these methods from the children classes.

# Important things to-do

- create a script for converting the ZStruct into a dataset (x,y). Take inspiration from utils/Ztools.py (ZStructTranslator.py) and from utils/TrainingUtils.py. Different from the current ZStructTranslator the new function, you can keep the same name, should take in input the Zstruct, the fields to exctract as a features, and the fields to exctract as output, and return a numpy vector (or similar) with the dataset.
- conversion of the (TrainingUtils) load_training_data, load_training_data_refit, load_training_data_multiday, load_training_data_simulated, load_training_data_simulated_multiday, load_training_data_auto functions for using the new ZStruct converted tool and structure. Check the load_training_data_multiday because it has some bugs in the current pybmi implementation. Ideally we would like to merge the train single and multi day into a single function. Each function should receive in input the list of days to use (1 or more).
- create scripts TrainDecoder (old TrainNetworkV4) and RigDecoderConnect (old Simulink_Connect_NHP_V2) that uses the new paradigm for decoder and dataset definitions but do the same things
- collection of function utilities for processing the dataset. These should be placed in a class called like (DatasetProcUtils or similar) and defined with static methods. Processing methods include normalization, smoothing, etc.. Keep track into a readme file with the list of all the methods included and briefly what they do.

# List of minor things to-do

- table of the decoders present in the repository. The table could include the name of the model, the parameters, a brief description and reference

