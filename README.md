# Description
This is a small demo for Featurecloud, showing how to calculate the mean in a federated manner.

# Algorithm
The algorithm simply shares the sum of all entries and the number of rows that were used for the sum with the coordinator. The coordinator adds the sums and divides by the total number of entries

# Input
A csv file with the column "salary"

# Output
A txt file with the mean value

# Creation
To create a similiar app, follow these steps.
Also, check out the [documentation of featurecloud](https://featurecloud.ai/assets/developer_documentation/index.html).

1. Initialise an empty app
```
featurecloud app new <yourappname>
```
2. Implement your logic in `states.py`
3. For any package you imported, state that in the requirements.txt so it will be loaded into the docker container
4. Build your docker image with
```
# Move out of the app folder
cd ..
# ls should now show the <yourappname> directory
featurecloud app build ./<yourappname> yourappname latest
# the arguments from this are [PATH] [IMAGE_NAME] [TAG]
# see the --help option for more info
```
5. Start the controller to be able to test your app
```
featurecloud controller start
```
This will create a data directory in the current working directory
In this data directory you can put folders for each clients input
for your simulations
6. Test/simulate your app either via the frondend (log in and under testing) or via the CLI (use `featurecloud test --help` for more info)


# Testing
1. featurecloud app build ./fedmean fedmean latest  

2. test start --client-dirs=.,.,.,. --generic-dir=.   --print-logs  --app-image fedmean


# original repo:
https://github.com/mdshihabullah/federated-predicted-euclidean-distance