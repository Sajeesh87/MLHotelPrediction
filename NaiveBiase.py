import pandas as pd

dataset_train = pd.read_csv('train.csv')
labelCount = {}
labelsProbability = {}

label_aggregation = dataset_train.groupby(dataset_train.hotel_market).size()
print(label_aggregation)

for label, label_count in label_aggregation.iteritems():
            labelCount[label] = label_count
            labelsProbability[label] = float(label_count + 1) / float(len(dataset_train) + 100)
            
probabilities = {}
for feature in list(dataset_train.columns.values):
    print(feature)
    probabilities[feature] = {}
    print ('Calculating probability for feature: {}'.format(feature))
    # iterate over all values for that feature
    for feature_value in dataset_train[feature].unique():
        probabilities[feature][feature_value] = {}
        # iterate over all class labels
        for class_label in labelCount:
            # count (feature=feature_value & class=class_value)
            feature_count = dataset_train[
                        (dataset_train[feature] == feature_value) &
                        (dataset_train.hotel_market == class_label)] \
                    .groupby(feature).size()
                    
            if not (len(feature_count) == 1):
                # print('feature: {}, value: {}, cluster: {}'.format(feature, feature_value, class_label))
                feature_count = 0
            else:
                feature_count = feature_count.iloc[0]
                
         # calculate probability (laplace correction)
            probability = float(feature_count + 1) / \
            float(labelCount[class_label] + len(labelCount))
            probabilities[feature][feature_value][class_label] = probability
print(probabilities)

dataset_test = pd.read_csv('train.csv')

columns = dataset_test.columns.values
predicted_labels = {}

# iterate through every row
for index, row in dataset_test.iterrows():
            max_prob = 0

            for class_label in labelsProbability:
                prob_product = 1

                for feature in columns:
                    feature_value = row[feature]
                    if (feature_value in probabilities[feature]):
                        prob_product *= probabilities[feature][feature_value][class_label]
                    else:
                        prob_product = 0

                # check if max prob, if so add to predicted_labels
                if prob_product > max_prob:
                    max_prob = prob_product
                    predicted_labels[index] = class_label

from pprint import pprint
pprint(predicted_labels)