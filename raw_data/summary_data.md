[NP] SOLart --> Sacar dos archivos diferentes
    s1:
        Accession
        Solubility 37
        Solubility 30

    Nota 1: El resto de archivos no tiene relevancia
    Nota 2: No se procesa porque son secuencias que se encuentran en prosol

[NP] optsolmut: Mutaciones binarias, armar mutaciones porque Nicole no quizo hacerlas y me hecha la culpa a mi 
    - Se ignora, ya que son solo 137 registros

[NP] NetSolP: Binario
    NESG: Leer y hacer merge
    PSI_biology: similar/idem a SoDoPe

[NP] SoDoPE -> Verlo David, solo, xq Nicole no tiene los códigos... retarla mañana viernes (28/06) :3
    - 2
    - 0
    - 1
    Nota: Ni NetSolP ni SoDoPE se procesan porque están incluidos dentro de otros set de datos
    
[OK] DeepMutSol: comparar con el PON-Sol2
    Se hacen la comparación con PON-SOL2 y se identifican mutaciones en reversa

[OK] mutsol: mutaciones con efectos 0, -1, y 1 y comparar con el all_protein de PON-Sol2
    Se hace la revisión y todas las mutaciones se encuentran en PON-SOL-2

[OK] PON-Sol2:
    Variantes mutacionales, armar la mutación
    0 -> Neutro
    1 -> Incrementa
    -1 -> Disminuye

[OK] GPSFun: leer datasets directos: los amo

[OK] GATSOL: porcentaje de solubilidad.... solo leer

[NP] eSOL: porcentaje y traer las secuencias desde GATSOL:
    No se parsea debido a que los datos se encuentran en en GATSOL.

[OK] EPSOL: leer los fastas y son solo valores binarios

[OK] DSRESsol: son binarios pero hay que transformarlos: 
    0,1,2 insoluble
    3,4 soluble

    NOTA: tiene también valores con 5

[OK] DeepSoluE: leer el fasta

[OK] DeepSol: mezclar los src con los tgt para obtener los datos

[OK] PLM-sol:Leer los fasta
    A-0: No soluble
    A-1: Soluble

[OK] PaRSnIP: Llegar y leer las cosas
    0: No soluble
    1: Soluble

[OK] Prosol: valor de solubilidad, puede ser que sea porcentaje
    sacar secuencia y numerito (Solubilidad con valor)

Notas Generales:

- Con respecto a las categóricas se crea un único set de datos
- Con respecto a las numéricas se trabaja solo con las de protsol debido a que no se parecen en nada los valores de las mediciones
- Con respecto a las categóricas por variante mutacional, se trabaja directamente con las secuencias procesadas desde PON-Sol2