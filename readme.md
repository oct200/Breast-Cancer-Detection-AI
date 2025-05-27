#  Definirea problemei

##  Ce se dă?
 Un set de imagini mamografice, fiecare însoțită de o etichetă ce indică natura tumorii. În plus, pot fi disponibile și informații adiționale precum localizarea exactă a formațiunii.

 Acces la modele de inteligență artificială din familia Transformer, pre-antrenate (ex: Vision Transformer – ViT), care pot fi adaptate ușor pentru sarcini specifice din domeniul imagisticii medicale.

 Un ecosistem software complet, construit în Python, cu librării moderne precum PyTorch și TensorFlow, plus unelte pentru prelucrarea imaginilor și evaluarea performanței modelelor AI.

##  Ce se cere?
Realizarea unui sistem de inteligență artificială capabil să analizeze automat imagini mamografice și să clasifice tumorile, contribuind la detectarea precoce a cancerului mamar.

Obiective specifice:

 Curățarea și preprocesarea imaginilor pentru a le face compatibile cu arhitecturile de rețele neuronale.

 Selectarea unui model Transformer potrivit și adaptarea acestuia (fine-tuning) pe setul de date furnizat.

 Antrenarea modelului și evaluarea performanței folosind metrici clinice esențiale: acuratețe, precizie, sensibilitate, specificitate și AUC-ROC.

 Minimizarea erorilor de clasificare, în special a falselor negative, care pot avea consecințe grave în contextul medical.

 Dezvoltarea unui pipeline automatizat, ușor de interpretat, care să ofere suport real medicilor în procesul decizional și să contribuie la diagnosticări mai rapide și mai sigure.

##  De ce e nevoie de AI pentru a rezolva problema?
Detectarea și clasificarea tumorilor mamare pe baza mamografiilor este o sarcină complexă, care implică analiza unor imagini cu detalii subtile, uneori greu de observat chiar și pentru specialiști cu experiență. În acest context, AI devine esențial din mai multe motive:

	1. Sensibilitate ridicată la detalii
Modelele de inteligență artificială, în special cele bazate pe rețele neuronale și arhitecturi Transformer, pot învăța să recunoască tipare fine sau anomalii care pot scăpa ochiului uman, mai ales în fazele incipiente ale bolii.

	2. Reducerea erorilor umane
Oboseala, subiectivitatea sau limitările resurselor pot duce la greșeli în interpretarea mamografiilor. AI poate oferi o analiză constantă, obiectivă și reproductibilă, reducând astfel riscul de erori de diagnostic, în special fals negative.

	3. Scalabilitate și eficiență
Un sistem automatizat poate analiza un volum mare de imagini într-un timp foarte scurt, sprijinind personalul medical în centre aglomerate sau în zone cu acces limitat la specialiști.

	4. Sprijin decizional, nu înlocuire
AI nu înlocuiește medicul radiolog, ci îl ajută prin furnizarea unei „a doua opinii” rapide și bine fundamentate, îmbunătățind calitatea deciziilor clinice.

	5. Îmbunătățirea diagnosticării timpurii
Cu cât cancerul este detectat mai devreme, cu atât cresc șansele de tratament eficient. AI contribuie direct la acest obiectiv, printr-o triere mai rapidă și precisă a cazurilor suspecte.

Pe scurt, AI oferă viteză, acuratețe și sprijin obiectiv, toate fiind factori esențiali în lupta împotriva cancerului de sân.

#  Analiza datelor de intrare

##  Ce tip de date avem?
Setul de date oferă imagini histopatologice digitale, utilizate pentru analiza și detectarea celulelor canceroase în țesutul mamar. Datele sunt caracterizate astfel:

   Tipul fișierelor:
	Imagini color (RGB) în format JPEG
	Dimensiunea fiecărei imagini este 50x50 pixeli, ceea ce le face potrivite pentru antrenarea rapidă a rețelelor neuronale
   
   Conținut:
	Fiecare imagine este un patch microscopic extras dintr-o lamă histologică scanată digital
	Aceste patch-uri provin din țesuturi obținute prin biopsie, colorate prin tehnici standard în histopatologie (ex: H&E staining)

   Etichete: 
	Datele sunt etichetate binar, în funcție de prezența sau absența unei tumori:
		0 – țesut sănătos
		1 – prezența IDC (Invasive Ductal Carcinoma), o formă agresivă de cancer mamar

   Tipul datelor:
	Date nestructurate: imaginile în sine, care necesită procesare vizuală avansată (deep learning) pentru a extrage informații relevante
	Date semi-structurate (prin etichete): fiecare imagine este asociată cu o clasă, ceea ce permite sarcini de învățare supervizată

##  Câte date avem?
Setul de date conține aproximativ ~280.000 imagini histopatologice digitale, obținute de la ~280 de pacienți. Fiecare imagine are dimensiunea de 96x96 pixeli și este etichetată pentru a indica prezența sau absența cancerului mamar (carcinom ductal invaziv - IDC).

	Număr total de imagini: 276.397

Acest volum mare de date oferă o bază solidă pentru antrenarea și validarea modelelor de inteligență artificială, permițând extragerea de caracteristici complexe și creșterea acurateții în clasificarea tumorilor.

##  Ce distribuție au datele?
Setul de date prezintă o distribuție dezechilibrată, în care numărul imaginilor cu țesut normal este mai mare comparativ cu cele care conțin tumori maligne (IDC). Acest dezechilibru poate influența procesul de antrenare al modelului, necesitând tehnici speciale pentru a preveni supraînvățarea pe clasa majoritară și pentru a asigura o clasificare corectă a ambelor categorii.

Clasificare:
	Imagini fără tumori (țesut normal): 197.721
	Imagini cu tumori (IDC prezente): 78.676

