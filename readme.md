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
Setul de date conține aproximativ ~280.000 imagini histopatologice digitale, obținute de la ~280 de pacienți. Fiecare imagine are dimensiunea de 50x50 pixeli și este etichetată pentru a indica prezența sau absența cancerului mamar (carcinom ductal invaziv - IDC).

	Număr total de imagini: 276.397

Acest volum mare de date oferă o bază solidă pentru antrenarea și validarea modelelor de inteligență artificială, permițând extragerea de caracteristici complexe și creșterea acurateții în clasificarea tumorilor.

##  Ce distribuție au datele?
Setul de date prezintă o distribuție dezechilibrată, în care numărul imaginilor cu țesut normal este mai mare comparativ cu cele care conțin tumori maligne (IDC). Acest dezechilibru poate influența procesul de antrenare al modelului, necesitând tehnici speciale pentru a preveni supraînvățarea pe clasa majoritară și pentru a asigura o clasificare corectă a ambelor categorii.

Clasificare:
	Imagini fără tumori (țesut normal): 197.721
	Imagini cu tumori (IDC prezente): 78.676

# Dezvoltarea unui model de AI si evaluarea performantei

## Ce arhitectura are modelul de AI?
Modelul de inteligență artificială utilizat este bazat pe arhitectura Transformer, care este potrivită pentru prelucrarea secvențială a datelor și este frecvent utilizată în sarcini precum clasificarea de text, analiza imaginilor sau generarea de limbaj natural. Această arhitectură folosește mecanisme de self-attention pentru a capta relațiile dintre elementele unei secvențe și permite procesarea în paralel a datelor.

## Ce setup (parametrii si hiper-parametrii) se folosesc pentru antrenarea si validarea modelului de AI?
Pentru antrenarea modelului Transformer, au fost folosiți următorii parametri și hiperparametri:

	1. Hiperparametri (configurați înainte de antrenare și care influențează procesul de învățare):
Funcția de pierdere (loss function): CrossEntropyLoss – potrivită pentru sarcini de clasificare multi-clasă.
Optimizator: Adam – un optimizator adaptiv eficient, configurat cu rata de învățare.
Learning rate (LR): valoarea variabilei LR – controlează pasul de actualizare al greutăților.
Numărul de epoci (EPOCHS): valoarea variabilei EPOCHS – numărul de treceri complete prin setul de date de antrenare.
Batch size (dimensiunea minibatch-ului): BATCH_SIZE – numărul de exemple folosite la fiecare pas de antrenare.
Sampler: sampler – controlează ordinea sau eșantionarea datelor din train_loader.

	2. Parametri dinamici monitorizați în timpul antrenării:
Loss: media pierderii pe întregul set de antrenare într-o epocă.
Accuracy: proporția de predicții corecte într-o epocă.

## Ce metrici de performanta se monitorizeaza?
Pentru evaluarea performanței modelului, sunt monitorizate următoarele metrice clasice de clasificare:

Accuracy – proporția totală de predicții corecte din toate exemplele;
Precision – proporția de exemple corect clasificate ca pozitive din totalul celor clasificate ca pozitive (utilă mai ales când false positive-urile sunt costisitoare);
Recall – proporția de exemple corect clasificate ca pozitive din totalul real al claselor pozitive (important când false negative-urile sunt critice);
Confusion matrix – o reprezentare tabelară care arată distribuția exactă a clasificărilor corecte și greșite pentru fiecare clasă, oferind o imagine de ansamblu asupra tipurilor de erori făcute de model.

# Propuneri de imbunatatiri
### Augmentare a imaginilor
Ajută la generalizare, mai ales când ai dezechilibru de clase.

	train_transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(10),
		transforms.ColorJitter(brightness=0.1, contrast=0.1),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize([0.5]*3, [0.5]*3)
	])

### Îmbunătățirea balansului între clase
Focal Loss în loc de CrossEntropyLoss, mai ales dacă avem clase dezechilibrate:

	class FocalLoss(nn.Module):
		def __init__(self, gamma=2.0):
			super().__init__()
			self.gamma = gamma
			self.ce = nn.CrossEntropyLoss()

		def forward(self, input, target):
			logp = self.ce(input, target)
			p = torch.exp(-logp)
			loss = (1 - p) ** self.gamma * logp
			return loss.mean()

	criterion = FocalLoss()

### Fine-tuning complet al modelului ViT
Deblocăm toate layerele, nu doar capul de clasificare:

	for param in model.parameters():
		param.requires_grad = True

### Learning rate scheduling
Learning rate scheduling este o tehnică prin care rata de învățare a modelului este ajustată automat pe parcursul antrenării, pentru a îmbunătăți convergența și performanța generală. Aceasta permite modelului să învețe mai rapid în etapele inițiale și mai fin în etapele finale, reducând riscul de overfitting.

### Validare și early stopping
Păstrăm modelul cu cele mai bune performanțe:

	best_acc = 0
	for epoch in range(EPOCHS):
		...
		if val_acc > best_acc:
			best_acc = val_acc
			torch.save(model.state_dict(), "best_model.pt")

### Postprocesare / Threshold customizat
În loc de argmax, folosim un prag adaptiv:

	probs = torch.softmax(output, dim=1)
	if probs[0, 1] > 0.7:
		print("Tumor detected")
	else:
		print("Likely normal")

### Evaluare cu matrice de confuzie, AUC, F1
Mai multe metrice pot ghida îmbunătățirile:

	from sklearn.metrics import classification_report, confusion_matrix
	
	y_true = [...]
	y_pred = [...]
	print(classification_report(y_true, y_pred, target_names=['no tumor', 'tumor']))

### Transfer learning de la modele antrenate pe imagini medicale (ex: ImageNet nu e ideal)
Căutăm modele pre-antrenate pe seturi medicale (ex: BioViT sau Google MedViT).