The folder is organized in 3 main subfolders:
    *training: Here are the training scripts. All of them perform the same task: They train with the model specified in the --state option.
    	       If you don't want to initialize the weights (in order to continue with a previous training), use the --skip-init option.
	       If you want to use another prototype, specify it in prototype.
    
    *states:   Here are python dictionaries which contain the parameters of each experiment.
    
    *decoding: Here are the scripts for using the trained models in an end-to-end translation task.

    *post-editing: Here are the scripts for using the library in an interactive simulated environment.

The script preprocess.sh takes a clean and tokenized corpus (in both source and target languages) and converts it into a valid format for Groundhog.