import os, sys

import pandas as pd

import numpy as np

from get_params import get_params

import matplotlib.pyplot as plt

import cv2



def display(params,query_id,ranking,relnotrel):



    ''' Display the first elements of the ranking '''





    # llegeix les imatges de consulta

    query_im =  cv2.imread(os.path.join(params['root'],params['database'],params['split'], 'images',query_id.split('.')[0] + '.jpg'))



    # com els fitxers tenen una terminacio .jpg em de trobar una manera per no contar aquesta terminacio. 

    if query_im is None:

        query_im =  cv2.imread(os.path.join(params['root'],params['database'],params['split'], 'images',query_id.split('.')[0] + '.JPG'))



    # utilitzem el contorn blau per la consulta



    query_im = cv2.cvtColor(query_im,cv2.COLOR_BGR2RGB)

    query_im = cv2.copyMakeBorder(query_im,100,100,100,100,cv2.BORDER_CONSTANT,value=[0,0,255])

    # inicialitza la figura

    fig = plt.figure(figsize=(20,10))

    ax = fig.add_subplot(4, 4, 1)



    # es visualitzen els resultats

    ax.imshow(query_im)

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)





    # mostrarem el 15 primers elements del ranking

    for i in range(15):



        #llegeix la imatge



        im =  cv2.imread(os.path.join(params['root'],params['database'],'train','images',ranking[0].tolist()[i] + '.jpg'))




        if im is None:



            im =  cv2.imread(os.path.join(params['root'],params['database'],'train', 'images',ranking[0].tolist()[i] + '.JPG'))



        # canviem la visualitzacio a RGB amb matplotlib

        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)



        # pintem el limits amb uns


        # si es selecciona correctament

        if relnotrel[i] == 1:

            # posem el contorn verd

            im = cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_CONSTANT,value= [0,255,0])



        # si no es compleix la condicio aleshores,

        else:

            # posem el contorn vermell

            im = cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_CONSTANT,value= [255,0,0])



        # mostrem la figura

        ax = fig.add_subplot(4, 4, i+2)

        ax.imshow(im)

        ax.axes.get_xaxis().set_visible(False)

        ax.axes.get_yaxis().set_visible(False)



    print "Displaying..."

    plt.show()



def read_annotation(params):



    # agafem les anotacions correctes

    annotation_val = pd.read_csv(os.path.join(params['root'],params['database'],params['split'],'annotation.txt'), sep='\t', header = 0)

    annotation_train = pd.read_csv(os.path.join(params['root'],params['database'],'train','annotation.txt'), sep='\t', header = 0)

    
    return annotation_val,annotation_train



def get_hitandmiss(ranking,query_class,annotation_train):


    # Inicialitzem la llista hit/miss 

    relnotrel = []



    # per a cada id d'imatge en el ranking

    for i in ranking[0].tolist():



        # agafem la clase de les anotacions

        i_class = list(annotation_train.loc[annotation_train['ImageID'] == i]['ClassID'])[0]



        # si coincideix amb alguna clase de la consulta

        if query_class == i_class:



            # significa que es correcta

            relnotrel.append(1)

        else:



            # si no es que em fallat

            relnotrel.append(0)



    return relnotrel



def AveragePrecision(relist):

    '''Takes a hit & miss list with ones and zeros and computes its average precision'''



    # inicialitzem la suma de precisions acumulada

    accu = 0



    # inicialitzem el nombre correcte de coincidencies trobades

    numRel = 0



    # per a tots el elements a la llista hit & miss 

    for k in range(len(relist)):



        # si el valor es 1

        if relist[k] == 1:



            # afegim un 1 al nombre de coincidencies correctes

            numRel = numRel + 1



            # calculem la precisio a K(+1 per que comencem 0) i ho acumulem

            accu += float( numRel )/ float(k+1)



    # quan em acavbat,dividim el nombre total de coincidencies rellevants, que es la suma dels 1 de la llista

    return (accu/np.sum(relist))



def load_ranking(params,query_id, annotation_val):



    ''' Loads and  returns the ranking from the txt. Returns the true class of the query image as well.'''



    # obtenim la verdadera clase de la validacio de l'imatge(totes les que em evaluat del ranking) 

    query_class = list(annotation_val.loc[annotation_val['ImageID'] == query_id.split('.')[0]]['ClassID'])[0]



    # obre el fitxer ranking

    ranking = pd.read_csv(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'],query_id.split('.')[0] + '.txt'),header= None)



    return query_class, ranking



#ha sortit del kaggle script

def save_ranking_file(file_to_save,image_id,ranking):



    '''

    :param file_to_save: name of the file to be saved

    :param image_id: name of the query image

    :param ranking: ranking for the image image_id

    :return: the updated state of the file to be saved

    '''



    # escriu un nom de consulta

    file_to_save.write(image_id.split('.')[0] + ',')



    # converteix l'element a un string i el posa a la llista ranking

    ranking = np.array(ranking).astype('str').tolist()



    # escriu un espai separat al ranking

    for item in ranking:

        file_to_save.write(item[0] + " ")



    file_to_save.write('\n')



    return file_to_save



def eval_rankings(params):



    ap_list = []



    # prepara per guardar el kaggle amb tots els rankings

    if params['save_for_kaggle']:



        file_to_save = open(os.path.join(params['root'],params['root_save'],params['kaggle_dir'],params['descriptor_type'] + '_' + params['split'] + '_ranking.csv'),'w')



        # escriu la primera linia amb la capçalera

        file_to_save.write("Query,RetrievedDocuments\n")



    # crea un diccionari per emmagatzemar el AP acumulat per cada clase

    dict_ = {key: 0 for key in params['possible_labels']}



    # agafa les anotacions correctes

    annotation_val, annotation_train = read_annotation(params)

    '''  

    # Used once to save ranking annotations for kaggle competition.

    gt_file_to_save = open(os.path.join(params['root'],params['root_save'],params['kaggle_dir'],params['split'] + '_rankingannotation.csv'),'w')

    kaggle_scripts.convert_ranking_annotation(annotation_val,annotation_train,gt_file_to_save)

  '''  



    # per a tots el ranking generats

    for val_id in os.listdir(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'])):



        query_class, ranking = load_ranking(params,val_id,annotation_val)



        # no evaluem les consultes de clases desconegudes

        if not query_class == "desconegut":

            #single_eval(params,val_id)  # activant aquesta linea fem un display per veure quines de les 10 primeres imatges s'han predit correctament i quines no

            if params['save_for_kaggle']:



                file_to_save = save_ranking_file(file_to_save,val_id,ranking)



            # obtenim la llista hit & miss 

            relnotrel = get_hitandmiss(ranking,query_class,annotation_train)



            # Calcula el promig de precisio de la llista 

            ap = AveragePrecision(relnotrel)





           
            dict_[query_class] += ap



            # guardem

            ap_list.append(ap)





    if params['save_for_kaggle']:



        file_to_save.close()



    return ap_list, dict_



def single_eval(params,query_id):



    # obtenim les anotacions correctes.

    annotation_val, annotation_train = read_annotation(params)



    # obtenim el ranking i les clases correctes de les consultes

    query_class, ranking = load_ranking(params,query_id,annotation_val)



    # aixi m'aseguro de que no agafo una imatge de clase desconeguda
    print query_class



    # el ranking el composem amb aquestes imatges. hauria de ser d'una mida de 450.

    print len(ranking)





    relnotrel = get_hitandmiss(ranking,query_class,annotation_train)



    display(params,query_id,ranking,relnotrel)



if __name__ == "__main__":



    params = get_params()



    ap_list, dict_ = eval_rankings(params)

    print 'Accuracy:'

    print 'Mean:',np.mean(ap_list)



    for id in dict_.keys():

        if not id == 'desconegut':

            # ho dividim per 10 ha que es el nombre d'iamatges per clase a la validacio

            print id+':', dict_[id]/10
