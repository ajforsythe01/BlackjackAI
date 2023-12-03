import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load data
totals = open("data\\totals.txt").readlines()
tags = open("data\\actions_log.txt").readlines()

# Added results so it can have another data point
results = open("data\\results.txt").readlines()

data_clean = []
tags_clean = []
results_clean = []
counter_data = 0

for result in results:
	result = result[:result.index('\n')]
	if result == "State.LOSE":
		results_clean = results_clean + [ 2 ]
	elif result == "State.WIN":
		results_clean = results_clean + [ 1 ]
	elif result == "State.TIE":
		results_clean = results_clean + [ 3 ]
	else:
		results_clean = results_clean + [ 0 ]
print(len(results_clean))

for datum in totals:
	clean_datum = datum[: datum.index('\n')].strip()
	clean_datum = clean_datum[1:-1].split(',')
	clean_datum[0] = int( clean_datum[0] )
	clean_datum[1] = int( clean_datum[1] )
	clean_datum[2] = int( clean_datum[2] )
	clean_datum.append( int(results_clean[counter_data]) )
	data_clean = data_clean + [ clean_datum ]
	counter_data = counter_data + 1
    

for tag in tags:
	tag = tag[:tag.index('\n')]
	if tag == "h":
		tags_clean = tags_clean + [ 1 ]
	else:
		tags_clean = tags_clean + [ 0 ]
print(len(tags_clean))

size = int( len(totals)*(0.75) )

train_data = np.array( data_clean[1:size] )
train_tags = np.array( tags_clean[1:size] )
test_data = np.array( data_clean[size:] )
test_tags = np.array( tags_clean[size:] )


model = keras.Sequential()
model.add( keras.layers.Dense( 4, input_dim=4 ) )
model.add( keras.layers.Dense( 16, activation = 'relu' ) )
model.add( keras.layers.Dense( 2, activation='softmax') )

model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

model.fit(train_data, train_tags, epochs=100)

test_loss, test_acc = model.evaluate(test_data, test_tags)

print('Test accuracy:', test_acc)

# Print Hard hand table
results = []

for k,i in enumerate(range(4,22)):
 	results = results + [ "" ]
 	for j in range(1,11):
 		prediction = model.predict( np.array([ [i,j, 0] ] ) )
 		if prediction[0][0] > prediction[0][1]:
 			results[k] = results[k] + "s"
 		else:  
 			results[k] = results[k] + "h"
            
print('Hard Hands')
print('---------------------------')
print( "  ", end="" )
for x in range( len(results[0]) ):
 	print( " " + str( (x+2)%10 ), end="" )
print( )
for i in range( len(results) ):
 	print( i+4, end="" )
 	if i+4 < 10:
 		print( "  ", end="" )
 	else:
 		print( "  ", end="" )
 	for j in range( len(results[i] ) ):
 		print( results[i][j], end=" " )

 	print( )
print('---------------------------') 

 
# Print Soft hand table
results = []

for k,i in enumerate(range(2,11)):
    results = results + [ "" ]
    for j in range(1,11):
        prediction = model.predict( np.array([ [i+11,j, 1] ] ) )
        if prediction[0][0] > prediction[0][1]:
            results[k] = results[k] + "s"
        else:
            results[k] = results[k] + "h"
print('Soft Hands')
print('---------------------------')
print( "  ", end="" )
for x in range( len(results[0]) ):
    print( " " + str( (x+2)%10 ), end="" )
print( )
for i in range( len(results) ):
    print( i+2, end="" )
    if i+2 < 10:
        print( "  ", end="" )
    else:
        print( " ", end="" )
    for j in range( len(results[i] ) ):
        print( results[i][j], end=" " )
    print( )
