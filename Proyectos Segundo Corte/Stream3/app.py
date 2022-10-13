import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl


df1= pd.read_csv("./data4.csv")
st.title("Proyecto 8")
st.subheader("Descripción suncita de las variables")
st.write(""" "INGLABO" = Ingresos Laborales """)
st.write(""" "P6050" = Variable discreta numerica que da información sobre ¿Cuál es el parentesco de ... con el jefe o jefa del hogar? a. Jefe (a) del hogar b. Pareja, esposo(a), cónyuge, compañero(a) c. Hijo(a), hijastro(a) d. Nieto(a) e. Otro pariente f. Empleado(a) del servicio doméstico y sus parientes g. Pensionista h.Trabajador i. Otro no pariente""")
st.write(""" "P6020" = Discreta numerica Información del Sexo del usuario 1 Hombre 2 Mujer """)
st.write(""" "P6040" = Variable Continua Numerica, ¿Cuántos años cumplidos tiene … ? Valores numericos en años, Si es menor de 1 año, escriba 00.""")
st.write(""" 'P6070' = Discreta numerica, Información del Estado Civil del Usuario;Actualmente: a. No esta casado(a) y vive en pareja hace menos de dos años b. No esta casado (a) y vive en pareja hace dos años o más c. Esta casado (a) d. Esta separado (a) o divorciado (a) e.Esta viudo (a) f. Esta soltero (a)""")
st.write(""" "ESC"= Continua Numerica, que da informacion de los Años de escolaridad """)
st.write(""" "P6426"=Variable Continua Numerica, respuesta a ¿cuanto tiempo lleva trabajando en esta empresa, negocio,industria, oficina, firma o finca de manera continua? se obtiene el valor en meses, si es menos de un mes el dato es 000""")
st.write(""" "P6430"=Discreta Numerica- Categorica se tiene información sobre el tipo de trabajo este puede ser:1. Obrero o empleado de empresa particular 2. Obrero o empleado del gobierno 3. Empleado doméstico 4. Trabajador por cuenta propia 5. Patrón o empleador 6. Trabajador familiar sin remuneración 7. Trabajador sin remuneración en empresas o negocios de otros hogares 8. Jornalero o peón 9.Otro""")
st.write(""" "P6800"=Variable Continua Numerica que nos da información sobre: ¿Cuántas horas a la semana trabaja normalmente.... en ese trabajo ? en Horas """)
st.write(""" "P6585S1"= Discreta Numerica información sobre si recibio el mes pasado subsidio o auxilio de alimentación 1 Sí 2 No 9 No sabe, no informa""")
st.write(""" "P6585S2"=Discreta Numerica información sobre si recibio el mes pasado subsidio o auxilio de transporte 1 Sí 2 No 9 No sabe, no informa""")
st.write(""" "P6585S3"= Discreta Numerica información sobre si recibio el mes pasado subsidio o auxilio familiar 1 Sí 2 No 9 No sabe, no informa""")
st.write(""" "P6585S4"=Discreta Numerica información sobre si recibio el mes pasado subsidio o auxilio de educación 1 Sí 2 No 9 No sabe, no informa""")

st.subheader("Gráficas relevantes")

st.image("./4/1.png")
st.image("./4/2.png")

st.write("""prediccion""")

P6020 = st.selectbox('Genero:',('Hombre','Mujer'))

P6020 = 1 if P6020 == 'Hombre' else 0

P6040 =st.number_input('Ingrese su Edad')

P6070 = st.selectbox('Estado Civil:',('Casado','Soltero'))
P6070 = 1 if P6070 == 'Casado' else 0

ESC = st.number_input('Ingrese sus años de escolaridad')
P6426 = st.number_input("Tiempo que lleva trabajando en su lugar de trabajo (en meses)")
P6800 = st.number_input("Tiempo que trabaja normalmente (en hora a la semana)")
P6585S1 = st.selectbox('Subsidio de alimentos:',('Si','No',"no sabe"))
P6585S1 = 1 if P6585S1 == 'Si' else 0

P6585S2 = st.selectbox('Subsidio de transporte:',('Si','No',"no sabe"))
P6585S2 = 1 if P6585S2 == 'Si' else 0
                                                  
P6585S3 = st.selectbox('Subsidio de familiar:',('Si','No',"no sabe"))
P6585S3 = 1 if P6585S3 == 'Si' else 0
                                                
P6585S4 = st.selectbox('Subsidio de auxilio:',('Si','No',"no sabe"))
P6585S4 = 1 if P6585S4 == 'Si' else 0
                                                 

st.subheader("""Modelo """)

rf_cls_pickle9= open('rf_clsproyecto9.pickle','rb')
rf_cls = pkl.load(rf_cls_pickle9)
print(rf_cls)

datos= [P6020,P6040,P6070,ESC,P6426,P6800,P6585S1,P6585S2,P6585S4]

prediction = rf_cls.predict([np.array(datos).reshape(1,-1)][0])[0]

resultado = presiction
st.write(resultado)
