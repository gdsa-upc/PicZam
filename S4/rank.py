
import os

import pickle

import numpy as np

from get_params import get_params

from sklearn.metrics.pairwise import pairwise_distances



def rank(params):



    # Carga el diccionaris de funcions de validacio

    val_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],

                             params['split'] + "_" + str(params['descriptor_size']) + "_"

                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p'),'rb'))



    train_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],

                             'train' + "_" + str(params['descriptor_size']) + "_"

                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p'),'rb'))





    # per a cada id d'imatge en el bloc validacio

    for val_id in val_features.keys():



        # agafem les caracteristiques

        bow_feats = val_features[val_id]



        # el ranking es composa amb els id de les imatges porcesades

        ranking = train_features.keys()



        X = np.array(train_features.values())



        # el .squeeze() redueix la dimensió de la formacio . Ex: transforma una matriu (400,1,100) a (400,100)

        distances = pairwise_distances(bow_feats,X.squeeze())





        # ordena el ranking d'acord amb les distancies. Ho convertim a numpy.array per ordenarlo, finalment ho enviem a la llista 

        ranking = list(np.array(ranking)[np.argsort(distances.squeeze())])



        # guardem el fitxer de text

        outfile = open(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'],val_id.split('.')[0] + '.txt'),'w')



        for item in ranking:



            outfile.write(item.split('.')[0] + '\n')



        outfile.close()



if __name__ == "__main__":



    params = get_params()

    rank(params)
