public class MultiLayerXOR {
	//giriş ile ara katman ağırlığı
	double w1[][] = { { 0.129952, 0.570345 }, { -0.923123, -0.328932 } }; 
	double w2[] = { 0.164732, 0.752621 }; // çıktı ile ara katman ağırlığı
	double b1[] = { 0.341332, -0.115223 }; // Giriş eşik değerleri
	double b2[] = { -0.993423 }; // Çıkış eşik değerleri
	double LR = 0.5; //öğrenme katsayısı
	double momentum = 0.8;
	int LH = 2; //ara katman sayısı
	int LO = 1; //çıkış katman sayısı
	int LI = 2; //giriş katman sayısı
	double output[] = new double[2];
	double net[] = new double[2];
	double dw1 [][]=new double [1][1];
	double dw2 []=new double[1];
	double db1[]=new double[1];
	double db2[]=new double[0];
	
	
	public double sigmoid(double x) { //sigmoid fonksiyonu  
		return (1 / (1 + Math.exp(x)));
	}

	public double sigmoid_derive(double x) { //sigmoid türevfonksiyonu
		return sigmoid(x) * (1 - sigmoid(x));
	}

	public double[] forward(double[][] input) { //ileri doğru hesaplama 
		for (int i = 0; i < LH; i++) {
			net[i] = 0.0;
			for (int j = 0; j < LI; j++) {
				net[i] += (input[i][j] * w1[i][j]);
			}
			net[i] += b1[i];
			net[i] = sigmoid(net[i]);

		}
		for (int i = 0; i < LO; i++) {

			for (int j = 0; j < LH; j++) {
				output[i] += (net[j] * w2[j]);
			}
			output[i] += b2[i];
			output[i] = sigmoid(output[i]);
		}

		return output;

	}





//geriye hesaplama
	public double backwards(double[] Output, double[][] inputs) { 
		double s2[] = new double[1];
		double s1[] = new double[2];

		for (int i = 0; i < LO; i++) { //çıktı ünitesi hatası
			s2[i] = sigmoid_derive(Output[i] * (Output[i] - output[i]));
		}

		for (int i = 0; i < LH; i++) { //ara katmandaki hata oranları
			double error = 0.0;
			for (int j = 0; j < LO; j++) {
				error += w2[j] * s2[j];

			}
			s1[i] = sigmoid_derive(net[i] * error);

		}
		//giriş ve arakatman değişim ağırlığı ve ara katman eşik değişimi
		for (int i = 0; i < LH; i++) { 
			for(int j=0; j<LI; j++) {
				dw1[j][i]+=LR*s1[i]*inputs[i][j]+momentum*dw1[j][i];
				
			}
			db1[i]+=LR*s1[i]+momentum*db1[i];
		}
		//çıktı ile ara katmanı değişim ağırlığı ve ara katman eşik değişimi
		for (int i = 0; i < LO; i++) {
			for(int j=0; j<LH; j++) {
				dw2[j]+=LR*s2[i]+momentum*dw2[j];
			}
			db2[i]+=LR*s2[i]+momentum*db2[i];
		}

		return s2[0];

	}
//yeni ağırlık ve eşik değerleri
	public void updateWeights (double LR) {
		for(int i=0; i<LO; i++) {
			for(int j=0; j<LH; j++) {
				w2[j]+=dw2[j];
				dw2[j]=0;
			}
			b2[i]+=db2[i];
			db2[i]=0;
		}
		for(int i=0; i<LH; i++) {
			for(int j=0; j<LI; j++) {
				w1[j][i]+=dw1[j][i];
				dw1[j][i]=0;
			}
			b1[i]+=db1[i];
			db1[i]=0;
	}
	}
}
