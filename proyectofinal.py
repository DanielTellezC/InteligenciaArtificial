import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from sklearn.cluster import KMeans

imagesNormal = [cv2.imread(file) for file in glob.glob("NORMAL_VISION/*.jpeg")]
imagesNeumonia = [cv2.imread(file) for file in glob.glob("PNEUMONIA_VISION/*.jpeg")]

def neumonia(img):
    ###Las imagenes se convierten a escala de grises para que sea más fácil el procesamiento###
    imagesNormalGrises = [cv2.cvtColor(file, cv2.COLOR_RGB2GRAY) for file in imagesNormal]
    imagesNeumoniaGrises = [cv2.cvtColor(file, cv2.COLOR_RGB2GRAY) for file in imagesNeumonia]

    ### Se hace una ecualización de imagenes para resaltar las partes afectadas#
    imagesNormalEcualizadas = [cv2.equalizeHist(file) for file in imagesNormalGrises]
    imagesNeumoniaEcualizadas = [cv2.equalizeHist(file) for file in imagesNeumoniaGrises]

    ### Se saca la media de cada imagen ###
    imagesNormalMedia = [[np.mean(file) for file in imagesNormalEcualizadas]]
    imagesNeumoniaMedia = [[np.mean(file) for file in imagesNeumoniaEcualizadas]]

    print(img)
    newimgmed = []
    if img < 5:
        newimgmed = imagesNormalMedia[0][40+img]
    elif img > 4 and img < 9:
        newimgmed = imagesNeumoniaMedia[0][40+(img-4)]


    arregloSumado = np.concatenate((imagesNormalMedia[0][:40] , imagesNeumoniaMedia[0][:40]), axis=0)

    li = np.array(arregloSumado).reshape(80, 1)
    ## metodo de k means
    k_means = KMeans(n_clusters=2)
    k_means.fit(li)

    ##centroides
    centroides = k_means.cluster_centers_
    etiquetas = k_means.labels_

    print("imagen",newimgmed)
    if newimgmed > min(li[etiquetas == 0]):
        print("con posible neumonia")
    elif newimgmed < max(li[etiquetas == 1]):
        print("Sin neumonia")

    print(newimgmed)
    plt.plot(li[etiquetas == 0], 'r.', label='Con posible neumonia')
    plt.plot(li[etiquetas == 1], 'b.', label='sin neumonia')
    plt.plot(newimgmed, 'g.', markersize=8 , label='imagen de entrada')
    plt.plot(centroides[0][0],'mo', markersize=8, label='Media')
    plt.plot(centroides[1][0],'mo', markersize=8)
    plt.legend(loc='best')
    plt.show()




    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    print("Ingrese un numero del 1 al 8, toma en cuenta que del 1 al 4 son imagenes sin neumonia y del 5 al 8 son imagenes con neumonia")
    num = input()
    num = int(num)
    neumonia(num)
    print(num)