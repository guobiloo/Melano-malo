# =============================================================================
# #--------------------------IMPORTS---------------------------------
# =============================================================================
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


# =============================================================================
# #############################################################################
# #############################################################################
# #------------------EVENTOS INTERACTIVOS DE USUARIO--------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
# =============================================================================
# obtener posicion de la imagen al hacer click
# =============================================================================
def mouse_clic(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        global punto
        punto = (x, y)
        print("click en pixel", punto)
        return punto


# =============================================================================
# #############################################################################
# #############################################################################
# #-------------------------------GRAFICOS---------------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
# =============================================================================
# cargar un conjunto de imagenes .jpg en un directorio
# =============================================================================
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder, filename))
        img = cv.cvtColor(img,cv.COLOR_RGB2BGR).astype(np.uint8)
        if img is not None:
            images.append(img)
    return np.array(images)


# =============================================================================
# calcular el histograma de un canal de la imagen (si e gris, no especificar canal)
# =============================================================================
def histograma(img,canal=0,mask=None,grafico='false',titulo=''):
        #obtener el histograma de un canal o de imagen en gris
        histr = cv.calcHist([img],[canal],mask,[256],[0,256])
        
        if(grafico=='true'):
            #normalizar histograma y plotear histograma
            plt.figure()
            h = np.copy(histr)
            cv.normalize(h,h,0,255,cv.NORM_MINMAX)
            plt.plot(h,'blue'),plt.title(titulo)
            plt.show()
        
        return histr


# =============================================================================
# graficar algo rapido en MatPlotLib
# =============================================================================
def graficar(img,titulo='',modo='gris',colormap='gray'):
    if(modo=='color'):
        plt.figure()
        plt.imshow(img[:,:,::-1]),plt.title(titulo)
        plt.show()
    if(modo=='gris'):
        plt.figure()
        plt.imshow(img,colormap),plt.title(titulo)
        plt.show()


# =============================================================================
# graficar planos RGB para analizar
# =============================================================================
def analizarRGB(img, flag='histograma',mask=None):
    img=np.uint8(np.copy(img))
    #graficar la imagen separada en cada plano de color
    b,g,r = cv.split(img) #separar imagen en sus planos de color
    plt.figure()
    plt.subplot(221),plt.imshow(img[:,:,::-1]),plt.title('img original')
    plt.subplot(222),plt.imshow(b,'gray'),plt.title('BLUE')
    plt.subplot(223),plt.imshow(g,'gray'),plt.title('GREEN')
    plt.subplot(224),plt.imshow(r,'gray'),plt.title('RED')
    plt.show()
   
    #analizar el histograma en rgb
    if(flag=='histograma'):
        color = ('blue','green','red')
        plt.figure()
        for i,col in enumerate(color):
            histr = cv.calcHist([img],[i],mask,[256],[0,256])
            h = np.copy(histr)
            cv.normalize(h,h,0,255,cv.NORM_MINMAX)
            plt.plot(h,color = col)
            plt.xlim([0,256])
        plt.title("histograma RGB")
        plt.show()
        

# =============================================================================
# pasar imagen RGB a gris y analizar
# =============================================================================
def analizarGRAY(img,flag='histograma',mask=None):
    img=np.uint8(np.copy(img))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    graficar(img,'imagen en gris')
   
    #analizar el histograma
    if(flag=='histograma'):
        plt.figure()
        histr = cv.calcHist([img],[0],mask,[256],[0,256])
        h = np.copy(histr)
        cv.normalize(h,h,0,255,cv.NORM_MINMAX)
        plt.plot(h,'blue')
        plt.xlim([0,256])
        plt.title("histograma de img GRIS")
        plt.show()
        
    return img
    

# =============================================================================
# graficar planos HSV para analizar
# =============================================================================  
def analizarHSV(img,flag='histograma',mask=None):
    img_patron = np.copy(img).astype(np.uint8)
    
    #pasar imagen a HSV
    img_patron = cv.cvtColor(img_patron, cv.COLOR_BGR2HSV) 
    H,S,V = cv.split(img_patron) #separar imagen en sus planos
    plt.figure()
    plt.subplot(131),plt.imshow(H,'gray'),plt.title('hue-tono')
    plt.subplot(132),plt.imshow(S,'gray'),plt.title('saturacion')
    plt.subplot(133),plt.imshow(V,'gray'),plt.title('value')
    plt.show()
    
    #analizar el histograma HSV
    if(flag=='histograma'):
        plt.figure()
        hist_h = cv.calcHist([img_patron],[0],mask,[180],[0,180])
        hist_s = cv.calcHist([img_patron],[1],mask,[256],[0,256])
        hist_v = cv.calcHist([img_patron],[2],mask,[256],[0,256])
        plt.subplot(131),plt.plot(hist_h),plt.title('hue-tono')
        plt.subplot(132),plt.plot(hist_s),plt.title('saturacion')
        plt.subplot(133),plt.plot(hist_v),plt.title('value')
        plt.show()
        


# =============================================================================
# #############################################################################
# #############################################################################
# #--------------------------------COLORES---------------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
# =============================================================================
# obtener un ROI de la imagen al hacer click en algun lugar
# =============================================================================
def getROI(img,alto,ancho):
    img_aux = np.copy(img)
    
    #definir ventana interactiva para hacer click
    winname = 'vnt'
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, mouse_clic)
    print("presione 'q' para terminar")
    while(1):
        cv.imshow(winname,img_aux)
        key = cv.waitKey(0)
        if key == ord("q"):
            break
    cv.destroyAllWindows()
    
    #graficar con rectangulo el area seleccionada
    cv.rectangle(img_aux,(punto[0]-int(ancho/2),punto[1]-int(alto/2)),(punto[0]+int(ancho/2),punto[1]+int(alto/2)),(255,255,255),1)
    graficar(img_aux,'muestra de imagen con roi seleccionado')
    
    #recortar ROI de la imagen
    x=punto[0]
    y=punto[1]
    if(np.size(img.shape)==3):
        roi = img[y-int(alto/2):y+int(alto/2) , x-int(ancho/2):x+int(ancho/2),:]
        print("roi de color generado")
    else:
        roi = img[y-int(alto/2):y+int(alto/2) , x-int(ancho/2):x+int(ancho/2)]
        print("roi de un solo plano generado")
        
    return roi


# =============================================================================
# generar mascara con pxs que esten dentro de un subespacio de color (ROI) de la imagen
# =============================================================================
def colorMask_rebanadoColor(img,roi,modo='rgb'):
    if(modo=='hsv'):
        hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
        hsvt = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        
        #calcular subespacio de color circular
        mediaH = np.mean(hsv[:,:,0])
        desvH = np.std(hsv[:,:,0])
        mediaS = np.mean(hsv[:,:,1])
        desvS = np.std(hsv[:,:,1])
        
        #establecer umbrales
        lower = np.array([
                mediaH-desvH])  
        upper = np.array([
                mediaH+desvH]) 
        lower2 = np.array([
                mediaS-desvS])  
        upper2 = np.array([
                mediaS+desvS])
        
        #enmascarar pixels cuyos valores esten dentro de la ROI de color
        h,s,v = cv.split(hsvt)
        hs=cv.merge((h,s))
        mask = cv.inRange(hs,np.array([lower,lower2]),np.array([upper,upper2])).astype(np.uint8)
        
    if(modo=='rgb'):
        #calcular subespacio de color circular
        centro_blue = np.mean(roi[:,:,0])
        centro_green = np.mean(roi[:,:,1])
        centro_red = np.mean(roi[:,:,2])
        desv_blue = np.std(roi[:,:,0])
        desv_green = np.std(roi[:,:,1])
        desv_red = np.std(roi[:,:,2])
        
        #establecer umbrales
        lower_muestra = np.array([
                centro_blue-desv_blue,
                centro_green-desv_green,
                centro_red-desv_red])
            
        upper_muestra = np.array([
                centro_blue+desv_blue,
                centro_green+desv_green,
                centro_red+desv_red     
                ])
        
        #enmascarar pixels cuyos valores esten dentro de la ROI de color
        mask = cv.inRange(img,lower_muestra,upper_muestra).astype(np.uint8)
    
    return mask


# =============================================================================
# generar mascara con un ROI considerando un solo plano de la imagen final.
# =============================================================================
def colorMask_rebanadoIndividual(img,roi,plano=0):
    #calcular subespacio de color circular
    if(np.size(roi.shape)==3):
        centro = np.mean(roi[:,:,plano])
        desv = np.std(roi[:,:,plano])
    else:
        centro = np.mean(roi)
        desv = np.std(roi)
    
    #establecer umbrales
    lower_muestra = centro-desv     
    upper_muestra = centro+desv
    
    #enmascarar pixels cuyos valores esten dentro de la ROI de color
    if(np.size(img.shape)==3):
        mask = cv.inRange(img[:,:,plano],lower_muestra,upper_muestra).astype(np.uint8)
    else:
        mask = cv.inRange(img,lower_muestra,upper_muestra).astype(np.uint8)
    
    return mask


# =============================================================================
# retroproyeccion es una version alternativa (a veces mejor) de rebanado de color por ROI
# =============================================================================
def colorMask_retroproyeccion(img,roi):
    hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
    hsvt = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    
    # calcula el histograma del roi (tono y saturacion)
    roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    
    # normaliza el histograma y aplica la retroproyección
    cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
    dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
    
    # Ahora aplica la covolución con un disco
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))
    cv.filter2D(dst,-1,disc,dst)
    
    # Aplica un umbral y convierte la imagen en blanco y negro
#    histograma(dst,grafico='true')
#    ret,thresh = cv.threshold(dst,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#    print("ret",ret)
    ret,thresh = cv.threshold(dst,3,255,cv.THRESH_BINARY)
    
    thresh = cv.merge((thresh,thresh,thresh))
    res = cv.bitwise_and(img,thresh)
    res = np.vstack((img,thresh,res))
    
    return res
        

# =============================================================================
# #############################################################################
# #############################################################################
# #--------------------------OP ARITMETICAS---------------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
# =============================================================================
# suma de imagenes en un solo canal
# =============================================================================
#ojo que las imagenes deben ser del mismo tamaño
def suma(imgs):
    cant_imagenes = imgs.shape[0]
    print("cantidad de ims: ",cant_imagenes)
    for i in range(cant_imagenes):
        imgs[i]=np.float64(imgs[i])

    accum = np.copy(imgs[0])
    
    for i in range(1,cant_imagenes):
        accum = accum + imgs[i]
    
    accum = accum / cant_imagenes
    accum[accum>255]=255
    
    return accum
    

# =============================================================================
# suma con peso entre 2 imagenes
# =============================================================================
#ojo que las imagenes deben ser del mismo tamaño
def suma_con_peso(img1,img2,p1,p2,constante=0):
    #la funcion de openCV solo funciona con imgs a color. Se agrega soporte a grises
    if(np.size(img1.shape)==2):
        img1=cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    if(np.size(img2.shape)==2):
        img2=cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
  
    #suma con pesos (blending). La constante se suma a cada valor una vez hecho el blending
    img_compuesta = cv.addWeighted(img1,p1,img2,p2,constante)
    return img_compuesta



# =============================================================================
# resta de 2 imagenes en un solo canal
# =============================================================================
#ojo que las imagenes deben ser del mismo tamaño
def resta(img1,img2):
    return cv.subtract(img1,img2)



# =============================================================================
# #############################################################################
# #############################################################################
# #--------------------------BORDE,CONTORNO,MASK---------------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
# =============================================================================
# generar un kernel simetrico rapido
# =============================================================================
def genQuickKernel(tam_kernel,tipo='rectangular'):
    if(tipo=='rectangulo'):
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(tam_kernel,tam_kernel))
    if(tipo=='circulo'):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(tam_kernel,tam_kernel))
    if(tipo=='cruz'):
        kernel = cv.getStructuringElement(cv.MORPH_CROSS,(tam_kernel,tam_kernel))
    return kernel


# =============================================================================
# construir una mascara binaria de un objeto a partir de sus bordes
# =============================================================================
def mask_bordes(img_bordes):
    mask = np.copy(img_bordes).astype(np.uint8)
    
    #corregir los bordes de la imagen por si tienen huecos (no corrige ruido sal)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    #encontrar la mascara
    imagen_cnt, contornos, jerarquia = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    mask=cv.drawContours(mask,contornos,-1,255,-1)
    
    return mask


# =============================================================================
# rellena un sector de una imagen en gris, especificando valores minimo y maximo de
# gris, un pixel semilla para indicar desde donde comenzar el analisis de vecinos,+
# y el color con el que se va llenando. Es imprecindible que una region sea homogenea
# y este separada de otras regiones homogeneas con bordes bien definidos
# =============================================================================
def floodFill(img,lowest_value,gretest_value,px_semilla,color,mascara=None):
    src = img.copy()
    
    connectivity = 8 #8-adyacencia
    flags = connectivity #considerar vecinos
    flags |= cv.FLOODFILL_FIXED_RANGE
    
    #floodFill(image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]]) -> retval, image, mask, rect
    cv.floodFill(src, mascara, px_semilla, color, lowest_value, gretest_value, flags)
    plt.figure()
    plt.imshow(src,'hot')
    plt.show()
    
    

# =============================================================================
# #############################################################################
# #############################################################################
# #------------------------------MORFOLOGIA---------------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
# =============================================================================
# operacion morfologica de apertura(erosion+dilatacion)
#    -trabaja sobre mascaras o contornos binarios
# =============================================================================
def apertura(img_binaria,kernel):
    img_resultado=np.copy(img_binaria).astype(np.uint8)
    img_resultado = cv.morphologyEx(img_resultado, cv.MORPH_OPEN, kernel)
    return img_resultado


# =============================================================================
# operacion morfologica de cierre(dilatacion+erosion)
# =============================================================================
def cierre(img_binaria,kernel):
    img_resultado=np.copy(img_binaria).astype(np.uint8)
    img_resultado = cv.morphologyEx(img_resultado, cv.MORPH_CLOSE, kernel)
    return img_resultado





