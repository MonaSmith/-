package net.mouse.line;



import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

import org.apache.spark.mllib.linalg.Vectors;
public class MouseLine {
	private final String inputPath = "/home/wfl/Documents/dsjtzs_txfz_training.txt";
	private final String inputPath2 = "/home/wfl/Documents/dsjtzs_txfz_test_combine.txt";
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
				MouseLine m = new MouseLine();
				m.mousePre();
			}catch(Exception e) {
				e.printStackTrace();
			}
	}
	
	
	
	private void mousePre() throws Exception
	{
		SparkConf conf = new SparkConf().setAppName("Mouse").setMaster("local[2]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> inp  = sc.textFile(inputPath);
        JavaRDD<String> tnp  = sc.textFile(inputPath2);
        JavaRDD<String> inf = inp.filter(f -> {
        	String[] arr = f.split(" ");
        	if(arr.length != 4)//训练集字段小于4为无效数据
        		return false;
        	else if((arr[1].split(";")).length < 2)//只有一个点的轨迹过滤掉
        		return false;
        	else
        		return true;
        });
        JavaRDD<String> tnf = tnp.filter(f -> {
        	String[] arr = f.split(" ");
        	if(arr.length != 3)//测试集小于3个字段的为无效数据
        		return false;
        	else if((arr[1].split(";")).length < 2)//只有一个点的轨迹过滤掉
        		return false;
        	else
        		return true;
        });
        JavaRDD<LabeledPoint> labeledPoint = trainingPre(inf);//训练集用map函数提取特征值返回的LabeledPoint
        JavaRDD<LabeledPoint> labeledPoint2 = testingPre(tnf);//测试集用map函数提取特征值返回的LabeledPoint
//      JavaRDD<LabeledPoint>[] d1= labeledPoint.randomSplit(new double[] {0.7,0.3}) ;
//      JavaRDD<LabeledPoint> traingData = d1[0];
//      JavaRDD<LabeledPoint> testingData = d1[1];
        int numClasses = 2;//类别数目
        Map<Integer,Integer> ca = new HashMap<>();
        int numTrees = 3;//决策树数目
        String featureSub = "all";//选取数据的方式
        String impurity = "gini";//纯度
        int maxDepth = 10;//决策树的深度
        int maxBins = 38;//广度
        int seed = 12345;//种子
        RandomForestModel mod = RandomForest.trainClassifier(labeledPoint, numClasses, ca, numTrees, featureSub,impurity,maxDepth
        		,maxBins,seed);//建立模型
        
        JavaPairRDD<Integer,Integer> predictionAndLabel = labeledPoint2.mapToPair(p ->{//预测测试集结果
        	int r = (int)(mod.predict(p.features()) + 0.5);
        	int l = (int)(p.label() + 0.5);
        	return new Tuple2<>(r,l);
        });
        predictionAndLabel.foreach(f ->{//遍历输出
        	System.out.println("cal:" + f._1()+",lab:" + f._2());
        });
//        double testErr = predictionAndLabel.filter(pl ->{
//        	if(pl._1() == pl._2())
//        		return false;
//        	else
//        		return true;
//        }).count() /(1.0* labeledPoint.count());
//        System.out.println("Error rating:" + testErr);
        
	
	}
	private JavaRDD<LabeledPoint> trainingPre(JavaRDD<String> inf) throws Exception{
      //提取特征值
       JavaRDD<LabeledPoint> labeledPoint = inf.map(f -> { 
        	double[] features = new double[18];
        	String[] arr = f.split(" ");
        	int label = Integer.valueOf(arr[3].trim());
        	double path = 0;
        	double timeSum = 0;
        	String[] arrPoint = arr[1].split(";");
        	double speed[] = new double[arrPoint.length-1];
        	for(int i = 0;i < arrPoint.length-1;i++)
        	{
        		String[] s = arrPoint[i].split(",");
        		String[] e = arrPoint[i + 1].split(",");       		
        		//整个路程的长度
        		
        		double le = Math.pow(Double.valueOf(e[0]) - Double.valueOf(s[0]), 2) +
        				Math.pow(Double.valueOf(e[1]) - Double.valueOf(s[1]), 2);
        		path += Math.sqrt(le);
        		//每一段的速度
        		speed[i] = le /  (Double.valueOf(e[2]) - Double.valueOf(s[2]));
        		//差分之和a
        		double reduce = Double.valueOf(e[2]) - Double.valueOf(s[2]);
        		timeSum += reduce;
        		
        	}
        	
        	//1.整个路程的平均速度
        	String[] s = arrPoint[0].split(",");
        	String[] e =arrPoint[arrPoint.length - 1].split(",");
        	double v = path / (Double.valueOf(e[2]) - Double.valueOf(s[2]));
        	features[0] = v;
        	//2.速度的方差
        	double sv=0;
        	for(int i = 0;i < speed.length;i++) {
        		double l =Math.pow(Double.valueOf(speed[i]) - v, 2);
        		sv += l;
        	}
        	double variance = sv / speed.length;
        	features[1] = variance;
        	//3.轨迹x方向走一半花的时间比 
        	double pathX = Double.valueOf(e[0]) - Double.valueOf(s[0]);
        	double middleX = pathX / 2;
        	double closeX = Math.abs(middleX - Double.valueOf(s[0]));
        	int index = 0;
        	for(int i = 0;i < arrPoint.length-1;i++) {
        		String[] p = arrPoint[i].split(",");
        		double x = Math.abs(middleX - Double.valueOf(p[0]));
        		if(x < closeX)
        		{
        			closeX = x;
        			index = i;
        		}
        	}
        	String[] m = arrPoint[index].split(",");
        	double x_ratio = (Double.valueOf(m[2]) - Double.valueOf(s[2])) / (Double.valueOf(e[2]) - Double.valueOf(s[2]));
        	features[2] = x_ratio;        	
        	//4.停顿时间占总时间
        	double timeCount = 0;
        	for(int i = 0;i < arrPoint.length-1;i++) {
        		String[] s1 = arrPoint[i].split(",");
        		String[] e1 = arrPoint[i + 1].split(",");
        		if(Double.valueOf(e1[0])== Double.valueOf(s1[0]) && Double.valueOf(e1[1])== Double.valueOf(s1[1])) {
        			timeCount += Double.valueOf(e1[2]) - Double.valueOf(s1[2]);
        		}
        	}
        	features[3] = timeCount/(Double.valueOf(e[2]) - Double.valueOf(s[2]));
        	//5.x坐标回退次数占总次数 
        	double back = 0;
        	for(int i = 0;i < arrPoint.length-1;i++) {
        		String[] s2 = arrPoint[i].split(",");
        		String[] e2 = arrPoint[i + 1].split(",");  
        		if(Double.valueOf(e2[0]) < Double.valueOf(s2[0])) {
        			back++;
        		}
        	}
        	features[4] = back/(arrPoint.length-1);
        	//6.时间轴t差分后的均值 
        	double ave = timeSum/(arrPoint.length-2);
        	features[5] = ave;
        	//7.t进行一阶差分后的标准差 
        	double sumRe = 0;
        	for(int i = 0;i < arrPoint.length-2;i++) {
        		String[] s3 = arrPoint[i].split(",");
        		String[] e3 = arrPoint[i + 1].split(","); 
        		double reduce = Double.valueOf(e3[2]) - Double.valueOf(s3[2]);
        		double reducePow = Math.pow(reduce - ave, 2);
        		sumRe += reducePow;
        	}
        	double stan = Math.sqrt(sumRe / (arrPoint.length-2));
        	features[6] = stan;  
        	//8.平均加速度
        	double sumA = 0;
        	for(int i = 0;i < arrPoint.length-2;i++) {
        		String[] s4 = arrPoint[i].split(",");
        		String[] e4 = arrPoint[i + 1].split(","); 
        		String[] f4 = arrPoint[i + 2].split(","); 
        		double le1 = Math.pow(Double.valueOf(e4[0]) - Double.valueOf(s4[0]), 2) +
        				Math.pow(Double.valueOf(e4[1]) - Double.valueOf(s4[1]), 2);
        		double le2 = Math.pow(Double.valueOf(f4[0]) - Double.valueOf(e4[0]), 2) +
        				Math.pow(Double.valueOf(f4[1]) - Double.valueOf(e4[1]), 2);
        		double v1 = le1 / (Double.valueOf(e4[2]) - Double.valueOf(s4[2]));
        		double v2 = le2 / (Double.valueOf(f4[2]) - Double.valueOf(e4[2]));
        		sumA += (v2-v1) / (Double.valueOf(f4[2]) - Double.valueOf(s4[2])); 
        	}
        	double avA = sumA / (arrPoint.length-2);
        	features[7] = avA;
        	//9.平均角度偏移
        	double offsets = 0;
        	for(int i = 0;i < arrPoint.length-2;i++) {
        		String[] s5 = arrPoint[i].split(",");
        		String[] e5 = arrPoint[i + 1].split(","); 
        		String[] f5 = arrPoint[i + 2].split(","); 
        		double PI = 3.1415926535897; 
        		double x1 = Double.valueOf(s5[0]) - Double.valueOf(e5[0]);
        		double y1 = Double.valueOf(s5[1]) - Double.valueOf(e5[1]);
        		double x2 = Double.valueOf(f5[0]) - Double.valueOf(e5[0]);
        		double y2 = Double.valueOf(f5[0]) - Double.valueOf(e5[0]);
        		double c = (x1 * x2) + (y1 * y2);
        		double z1 = Math.sqrt(x1*x1+y1*y1);
        		double z2 = Math.sqrt(x2*x2+y2*y2);
        		double A = c/(z1*z2);
        		double offset = Math.acos(A)*180/PI;
        		offsets += offset;	 
        	}
        	double aveOffset = offsets / (arrPoint.length-2);
        	features[8] = aveOffset;
        	//10.平均角度偏移标准差
        	double sumOff = 0;
        	for(int i = 0;i < arrPoint.length-2;i++) {
        		String[] s6 = arrPoint[i].split(",");
        		String[] e6 = arrPoint[i + 1].split(","); 
        		String[] f6 = arrPoint[i + 2].split(","); 
        		double PI_A = 3.1415926535897; 
        		double x3 = Double.valueOf(s6[0]) - Double.valueOf(e6[0]);
        		double y3 = Double.valueOf(s6[1]) - Double.valueOf(e6[1]);
        		double x4 = Double.valueOf(f6[0]) - Double.valueOf(e6[0]);
        		double y4 = Double.valueOf(f6[0]) - Double.valueOf(e6[0]);
        		double c1 = (x3 * x4) + (y3 * y4);
        		double z3 = Math.sqrt(x3*x3+y3*y3);
        		double z4 = Math.sqrt(x4*x4+y4*y4);
        		double A1 = c1/(z3*z4);
        		double offset = Math.acos(A1)*180/PI_A;
        		double subPow = Math.pow(offset - aveOffset, 2);
        		sumOff += subPow;
        	}
        	double offStand = Math.sqrt(sumOff / (arrPoint.length-2));
        	features[9] = offStand;
        	//11.速度的一阶差分平均值
        	double sumSub = 0;
        	for(int i = 0;i < speed.length-1;i++)
        	{
        		double sub = speed[i+1] - speed[i];
        		sumSub += sub;       		
        	}
        	double aveSub = sumSub / (speed.length-1);
        	features[10] = aveSub;       	
        	//12.同一时间坐标是否变化
        	double isStop = 0;
        	for(int i = 0;i < arrPoint.length-1;i++) {
        		String[] s7 = arrPoint[i].split(",");
        		String[] e7 = arrPoint[i + 1].split(",");
        		if((Double.valueOf(s7[0])!=Double.valueOf(e7[0])||Double.valueOf(s7[1])!=Double.valueOf(e7[1]))
        				&&Double.valueOf(s7[2])==Double.valueOf(e7[2])) {
        			isStop = 1;
        		}
        	}
        	features[11] = isStop;    
        	//13.走一半路花的时间占总时间比
        	double half_path = path/2;
        	double path2 = 0;
        	double min_diff_path = 100000;
        	int index2 = 0;
        	for(int i=0;i< arrPoint.length-1;i++)
        	{
        		String[] s8 = arrPoint[i].split(",");
        		String[] e8 = arrPoint[i + 1].split(",");       		
        		
        		
        		double l = Math.pow(Double.valueOf(e8[0]) - Double.valueOf(s8[0]), 2) +
        				Math.pow(Double.valueOf(e8[1]) - Double.valueOf(s8[1]), 2);
        		path2 += Math.sqrt(l);
        		if(Math.abs(half_path - path2) < min_diff_path) {
    				min_diff_path = Math.abs(half_path-path2);
    				index = i;
    			}

        	}
        	String[] n = arrPoint[index].split(",");
        	double timehalf_ratio = (Double.valueOf(n[2]) - Double.valueOf(s[2])) / (Double.valueOf(e[2]) - Double.valueOf(s[2]));
        	 
        	features[12] = timehalf_ratio; 
        	//14.最大加速度
        	double maxA = 0;
        	for(int i = 0;i < arrPoint.length-2;i++) {
           		String[] s9 = arrPoint[i].split(",");
           		String[] e9 = arrPoint[i + 1].split(","); 
           		String[] f9 = arrPoint[i + 2].split(","); 
           		double le1 = Math.pow(Double.valueOf(e9[0]) - Double.valueOf(s9[0]), 2) +
           				Math.pow(Double.valueOf(e9[1]) - Double.valueOf(s9[1]), 2);
           		double le2 = Math.pow(Double.valueOf(f9[0]) - Double.valueOf(e9[0]), 2) +
           				Math.pow(Double.valueOf(f9[1]) - Double.valueOf(e9[1]), 2);
           		double v1 = le1 / (Double.valueOf(e9[2]) - Double.valueOf(s9[2]));
           		double v2 = le2 / (Double.valueOf(f9[2]) - Double.valueOf(e9[2]));
           		double a1 = (v2-v1) / (Double.valueOf(f9[2]) - Double.valueOf(s9[2]));  
           		if(a1 > maxA)
           		{
           			maxA = a1;
           		}
           	}
        	features[13] = maxA; 
        	//15.是否停顿
        	double stop = 0;
        	for(int i = 0;i < arrPoint.length-1;i++) {
        		String[] s10 = arrPoint[i].split(",");
        		String[] e10 = arrPoint[i + 1].split(",");
        		if(Double.valueOf(e10[0])== Double.valueOf(s10[0]) && Double.valueOf(e10[1])== Double.valueOf(s10[1])) {
        			stop = 1;
        		}
        	}
        	features[14] = stop;
        	//16.最大速度
        	double max_speed = 0;
        	for(int i=0;i<speed.length;i++)
        	{
        		if(speed[i] > max_speed) {
        			max_speed = speed[i];
        		}
        	}
        	features[15] = max_speed;
        	//17.最大角度
        	double maxM = 0;
           	for(int i = 0;i < arrPoint.length-2;i++) {
           		String[] s11 = arrPoint[i].split(",");
           		String[] e11 = arrPoint[i + 1].split(","); 
           		String[] f11 = arrPoint[i + 2].split(","); 
           		double PI_A1 = 3.1415926535897; 
           		double x5 = Double.valueOf(s11[0]) - Double.valueOf(e11[0]);
           		double y5 = Double.valueOf(s11[1]) - Double.valueOf(e11[1]);
           		double x6 = Double.valueOf(f11[0]) - Double.valueOf(e11[0]);
           		double y6 = Double.valueOf(f11[0]) - Double.valueOf(e11[0]);
           		double c2 = (x5 * x6) + (y5 * y6);
           		double z5 = Math.sqrt(x5*x5+y5*y5);
           		double z6 = Math.sqrt(x6*x6+y6*y6);
           		double A2 = c2/(z5*z6);
           		double offset1 = Math.acos(A2)*180/PI_A1;
           		if(offset1 > maxM) {
           			maxM = offset1;
           		}
           	}
           	features[16] = maxM;
        	//18.相邻点构成的角度序列一阶差分后的均值
           	double aveFirst = 0;
           	double sunFirst = 0;
           	for(int i = 0;i < arrPoint.length-3;i++) {
           		String[] s12 = arrPoint[i].split(",");
           		String[] e12 = arrPoint[i + 1].split(","); 
           		String[] f12 = arrPoint[i + 2].split(","); 
           		double PI_A2 = 3.1415926535897; 
           		double x7 = Double.valueOf(s12[0]) - Double.valueOf(e12[0]);
           		double y7 = Double.valueOf(s12[1]) - Double.valueOf(e12[1]);
           		double x8 = Double.valueOf(f12[0]) - Double.valueOf(e12[0]);
           		double y8 = Double.valueOf(f12[0]) - Double.valueOf(e12[0]);
           		double c3 = (x7 * x8) + (y7 * y8);
           		double z7 = Math.sqrt(x7*x7+y7*y7);
           		double z8 = Math.sqrt(x8*x8+y8*y8);
           		double A3 = c3/(z7*z8);
           		double offset2 = Math.acos(A3)*180/PI_A2;
           		
           		String[] s13 = arrPoint[i+1].split(",");
           		String[] e13 = arrPoint[i + 2].split(","); 
           		String[] f13 = arrPoint[i + 3].split(","); 
           		double PI_A3 = 3.1415926535897; 
           		double x9 = Double.valueOf(s13[0]) - Double.valueOf(e13[0]);
           		double y9 = Double.valueOf(s13[1]) - Double.valueOf(e13[1]);
           		double x10 = Double.valueOf(f13[0]) - Double.valueOf(e13[0]);
           		double y10 = Double.valueOf(f13[0]) - Double.valueOf(e13[0]);
           		double c4 = (x9 * x10) + (y9 * y10);
           		double z9 = Math.sqrt(x9*x9+y9*y9);
           		double z10 = Math.sqrt(x10*x10+y10*y10);
           		double A4 = c4/(z9*z10);
           		double offset3 = Math.acos(A4)*180/PI_A3;
           		
           		sunFirst += (offset3 - offset2);
           	}
           	aveFirst = sunFirst/ arrPoint.length-3;
           	features[17] = aveFirst;
           	
        	return new LabeledPoint(label,Vectors.dense(features));
        });
       return labeledPoint;
       
	}
	private JavaRDD<LabeledPoint> testingPre(JavaRDD<String> tnf) throws Exception{
       JavaRDD<LabeledPoint> labeledPoint2 = tnf.map(f -> { 
       	double[] features = new double[18];
       	String[] arr = f.split(" ");
       	int label = Integer.valueOf(arr[0].trim());
       	double path = 0;
       	double timeSum = 0;
       	String[] arrPoint = arr[1].split(";");
       	double speed[] = new double[arrPoint.length-1];
       	for(int i = 0;i < arrPoint.length-1;i++)
       	{
       		String[] s = arrPoint[i].split(",");
       		String[] e = arrPoint[i + 1].split(",");       		
       		//整个路程的长度
       		
       		double le = Math.pow(Double.valueOf(e[0]) - Double.valueOf(s[0]), 2) +
       				Math.pow(Double.valueOf(e[1]) - Double.valueOf(s[1]), 2);
       		path += Math.sqrt(le);
       		//每一段的速度
       		speed[i] = le /  (Double.valueOf(e[2]) - Double.valueOf(s[2]));
       		//差分之和
       		double reduce = Double.valueOf(e[2]) - Double.valueOf(s[2]);
       		timeSum += reduce;
       		
       	}
       	
       	//1.整个路程的平均速度
       	String[] s = arrPoint[0].split(",");
       	String[] e =arrPoint[arrPoint.length - 1].split(",");
       	double v = path / (Double.valueOf(e[2]) - Double.valueOf(s[2]));
       	features[0] = v;
       	//2.速度的方差
       	double sv=0;
       	for(int i = 0;i < speed.length;i++) {
       		double l =Math.pow(Double.valueOf(speed[i]) - v, 2);
       		sv += l;
       	}
       	double variance = sv / speed.length;
       	features[1] = variance;
       	//3.轨迹x方向走一半花的时间比 
       	double pathX = Double.valueOf(e[0]) - Double.valueOf(s[0]);
       	double middleX = pathX / 2;
       	double closeX = Math.abs(middleX - Double.valueOf(s[0]));
       	int index = 0;
       	for(int i = 0;i < arrPoint.length-1;i++) {
       		String[] p = arrPoint[i].split(",");
       		double x = Math.abs(middleX - Double.valueOf(p[0]));
       		if(x < closeX)
       		{
       			closeX = x;
       			index = i;
       		}
       	}
       	String[] m = arrPoint[index].split(",");
       	double x_ratio = (Double.valueOf(m[2]) - Double.valueOf(s[2])) / (Double.valueOf(e[2]) - Double.valueOf(s[2]));
       	features[2] = x_ratio;        	
       	//4.停顿时间占总时间
       	double timeCount = 0;
       	for(int i = 0;i < arrPoint.length-1;i++) {
       		String[] s1 = arrPoint[i].split(",");
       		String[] e1 = arrPoint[i + 1].split(",");
       		if(Double.valueOf(e1[0])== Double.valueOf(s1[0]) && Double.valueOf(e1[1])== Double.valueOf(s1[1])) {
       			timeCount += Double.valueOf(e1[2]) - Double.valueOf(s1[2]);
       		}
       	}
       	features[3] = timeCount/(Double.valueOf(e[2]) - Double.valueOf(s[2]));
       	//5.x坐标回退次数占总次数 
       	double back = 0;
       	for(int i = 0;i < arrPoint.length-1;i++) {
       		String[] s2 = arrPoint[i].split(",");
       		String[] e2 = arrPoint[i + 1].split(",");  
       		if(Double.valueOf(e2[0]) < Double.valueOf(s2[0])) {
       			back++;
       		}
       	}
       	features[4] = back/(arrPoint.length-1);
       	//6.时间轴t差分后的均值 
       	double ave = timeSum/(arrPoint.length-2);
       	features[5] = ave;
       	//7.t进行一阶差分后的标准差 
       	double sumRe = 0;
       	for(int i = 0;i < arrPoint.length-2;i++) {
       		String[] s3 = arrPoint[i].split(",");
       		String[] e3 = arrPoint[i + 1].split(","); 
       		double reduce = Double.valueOf(e3[2]) - Double.valueOf(s3[2]);
       		double reducePow = Math.pow(reduce - ave, 2);
       		sumRe += reducePow;
       	}
       	double stan = Math.sqrt(sumRe / (arrPoint.length-2));
       	features[6] = stan;  
       	//8.平均加速度
       	double sumA = 0;
       	for(int i = 0;i < arrPoint.length-2;i++) {
       		String[] s4 = arrPoint[i].split(",");
       		String[] e4 = arrPoint[i + 1].split(","); 
       		String[] f4 = arrPoint[i + 2].split(","); 
       		double le1 = Math.pow(Double.valueOf(e4[0]) - Double.valueOf(s4[0]), 2) +
       				Math.pow(Double.valueOf(e4[1]) - Double.valueOf(s4[1]), 2);
       		double le2 = Math.pow(Double.valueOf(f4[0]) - Double.valueOf(e4[0]), 2) +
       				Math.pow(Double.valueOf(f4[1]) - Double.valueOf(e4[1]), 2);
       		double v1 = le1 / (Double.valueOf(e4[2]) - Double.valueOf(s4[2]));
       		double v2 = le2 / (Double.valueOf(f4[2]) - Double.valueOf(e4[2]));
       		sumA += (v2-v1) / (Double.valueOf(f4[2]) - Double.valueOf(s4[2])); 
       	}
       	double avA = sumA / (arrPoint.length-2);
       	features[7] = avA;
       	//9.平均角度偏移和
       	double offsets = 0;
       	for(int i = 0;i < arrPoint.length-2;i++) {
       		String[] s5 = arrPoint[i].split(",");
       		String[] e5 = arrPoint[i + 1].split(","); 
       		String[] f5 = arrPoint[i + 2].split(","); 
       		double PI = 3.1415926535897; 
       		double x1 = Double.valueOf(s5[0]) - Double.valueOf(e5[0]);
       		double y1 = Double.valueOf(s5[1]) - Double.valueOf(e5[1]);
       		double x2 = Double.valueOf(f5[0]) - Double.valueOf(e5[0]);
       		double y2 = Double.valueOf(f5[0]) - Double.valueOf(e5[0]);
       		double c = (x1 * x2) + (y1 * y2);
       		double z1 = Math.sqrt(x1*x1+y1*y1);
       		double z2 = Math.sqrt(x2*x2+y2*y2);
       		double A = c/(z1*z2);
       		double offset = Math.acos(A)*180/PI;
       		offsets += offset;	 
       	}
       	double aveOffset = offsets / (arrPoint.length-2);
       	features[8] = aveOffset;
       	//10.平均角度偏移标准差
       	double sumOff = 0;
       	for(int i = 0;i < arrPoint.length-2;i++) {
       		String[] s6 = arrPoint[i].split(",");
       		String[] e6 = arrPoint[i + 1].split(","); 
       		String[] f6 = arrPoint[i + 2].split(","); 
       		double PI_A = 3.1415926535897; 
       		double x3 = Double.valueOf(s6[0]) - Double.valueOf(e6[0]);
       		double y3 = Double.valueOf(s6[1]) - Double.valueOf(e6[1]);
       		double x4 = Double.valueOf(f6[0]) - Double.valueOf(e6[0]);
       		double y4 = Double.valueOf(f6[0]) - Double.valueOf(e6[0]);
       		double c1 = (x3 * x4) + (y3 * y4);
       		double z3 = Math.sqrt(x3*x3+y3*y3);
       		double z4 = Math.sqrt(x4*x4+y4*y4);
       		double A1 = c1/(z3*z4);
       		double offset = Math.acos(A1)*180/PI_A;
       		double subPow = Math.pow(offset - aveOffset, 2);
       		sumOff += subPow;
       	}
       	double offStand = Math.sqrt(sumOff / (arrPoint.length-2));
       	features[9] = offStand;
       	//11.速度的一阶差分平均值
       	double sumSub = 0;
       	for(int i = 0;i < speed.length-1;i++)
       	{
       		double sub = speed[i+1] - speed[i];
       		sumSub += sub;       		
       	}
       	double aveSub = sumSub / (speed.length-1);
       	features[10] = aveSub;       	
       	//12.同一时间坐标是否变化
       	double isStop = 0;
    	for(int i = 0;i < arrPoint.length-1;i++) {
    		String[] s7 = arrPoint[i].split(",");
    		String[] e7 = arrPoint[i + 1].split(",");
    		if((Double.valueOf(s7[0])!=Double.valueOf(e7[0])||Double.valueOf(s7[1])!=Double.valueOf(e7[1]))
    				&&Double.valueOf(s7[2])==Double.valueOf(e7[2])) {
    			isStop = 1;
    		}
    	}
    	features[11] = isStop; 
    	//13.走一半路花的时间占总时间比
    	double half_path = path/2;
    	double path2 = 0;
    	double min_diff_path = 100000;
    	int index2 = 0;
    	for(int i=0;i< arrPoint.length-1;i++)
    	{
    		String[] s8 = arrPoint[i].split(",");
    		String[] e8 = arrPoint[i + 1].split(",");       		
    		
    		
    		double l = Math.pow(Double.valueOf(e8[0]) - Double.valueOf(s8[0]), 2) +
    				Math.pow(Double.valueOf(e8[1]) - Double.valueOf(s8[1]), 2);
    		path2 += Math.sqrt(l);
    		if(Math.abs(half_path - path2) < min_diff_path) {
				min_diff_path = Math.abs(half_path-path2);
				index = i;
			}

    	}
    	String[] n = arrPoint[index].split(",");
    	double timehalf_ratio = (Double.valueOf(n[2]) - Double.valueOf(s[2])) / (Double.valueOf(e[2]) - Double.valueOf(s[2]));
    	 
    	features[12] = timehalf_ratio; 
       	//14.最大加速度
    	double maxA = 0;
    	for(int i = 0;i < arrPoint.length-2;i++) {
       		String[] s9 = arrPoint[i].split(",");
       		String[] e9 = arrPoint[i + 1].split(","); 
       		String[] f9 = arrPoint[i + 2].split(","); 
       		double le1 = Math.pow(Double.valueOf(e9[0]) - Double.valueOf(s9[0]), 2) +
       				Math.pow(Double.valueOf(e9[1]) - Double.valueOf(s9[1]), 2);
       		double le2 = Math.pow(Double.valueOf(f9[0]) - Double.valueOf(e9[0]), 2) +
       				Math.pow(Double.valueOf(f9[1]) - Double.valueOf(e9[1]), 2);
       		double v1 = le1 / (Double.valueOf(e9[2]) - Double.valueOf(s9[2]));
       		double v2 = le2 / (Double.valueOf(f9[2]) - Double.valueOf(e9[2]));
       		double a1 = (v2-v1) / (Double.valueOf(f9[2]) - Double.valueOf(s9[2]));  
       		if(a1 > maxA)
       		{
       			maxA = a1;
       		}
       	}
    	features[13] = maxA; 
    	//15.是否停顿
    	double stop = 0;
    	for(int i = 0;i < arrPoint.length-1;i++) {
    		String[] s10 = arrPoint[i].split(",");
    		String[] e10 = arrPoint[i + 1].split(",");
    		if(Double.valueOf(e10[0])== Double.valueOf(s10[0]) && Double.valueOf(e10[1])== Double.valueOf(s10[1])) {
    			stop = 1;
    		}
    	}
    	features[14] = stop;
    	//16.最大速度
    	double max_speed = 0;
    	for(int i=0;i<speed.length;i++)
    	{
    		if(speed[i] > max_speed) {
    			max_speed = speed[i];
    		}
    	}
    	features[15] = max_speed;
    	//17.最大角度
    	double maxM = 0;
       	for(int i = 0;i < arrPoint.length-2;i++) {
       		String[] s11 = arrPoint[i].split(",");
       		String[] e11 = arrPoint[i + 1].split(","); 
       		String[] f11 = arrPoint[i + 2].split(","); 
       		double PI_A1 = 3.1415926535897; 
       		double x5 = Double.valueOf(s11[0]) - Double.valueOf(e11[0]);
       		double y5 = Double.valueOf(s11[1]) - Double.valueOf(e11[1]);
       		double x6 = Double.valueOf(f11[0]) - Double.valueOf(e11[0]);
       		double y6 = Double.valueOf(f11[0]) - Double.valueOf(e11[0]);
       		double c2 = (x5 * x6) + (y5 * y6);
       		double z5 = Math.sqrt(x5*x5+y5*y5);
       		double z6 = Math.sqrt(x6*x6+y6*y6);
       		double A2 = c2/(z5*z6);
       		double offset1 = Math.acos(A2)*180/PI_A1;
       		if(offset1 > maxM) {
       			maxM = offset1;
       		}
       	}
       	features[16] = maxM;
      //18.相邻点构成的角度序列一阶差分后的均值
       	double aveFirst = 0;
       	double sunFirst = 0;
       	for(int i = 0;i < arrPoint.length-3;i++) {
       		String[] s12 = arrPoint[i].split(",");
       		String[] e12 = arrPoint[i + 1].split(","); 
       		String[] f12 = arrPoint[i + 2].split(","); 
       		double PI_A2 = 3.1415926535897; 
       		double x7 = Double.valueOf(s12[0]) - Double.valueOf(e12[0]);
       		double y7 = Double.valueOf(s12[1]) - Double.valueOf(e12[1]);
       		double x8 = Double.valueOf(f12[0]) - Double.valueOf(e12[0]);
       		double y8 = Double.valueOf(f12[0]) - Double.valueOf(e12[0]);
       		double c3 = (x7 * x8) + (y7 * y8);
       		double z7 = Math.sqrt(x7*x7+y7*y7);
       		double z8 = Math.sqrt(x8*x8+y8*y8);
       		double A3 = c3/(z7*z8);
       		double offset2 = Math.acos(A3)*180/PI_A2;
       		
       		String[] s13 = arrPoint[i+1].split(",");
       		String[] e13 = arrPoint[i + 2].split(","); 
       		String[] f13 = arrPoint[i + 3].split(","); 
       		double PI_A3 = 3.1415926535897; 
       		double x9 = Double.valueOf(s13[0]) - Double.valueOf(e13[0]);
       		double y9 = Double.valueOf(s13[1]) - Double.valueOf(e13[1]);
       		double x10 = Double.valueOf(f13[0]) - Double.valueOf(e13[0]);
       		double y10 = Double.valueOf(f13[0]) - Double.valueOf(e13[0]);
       		double c4 = (x9 * x10) + (y9 * y10);
       		double z9 = Math.sqrt(x9*x9+y9*y9);
       		double z10 = Math.sqrt(x10*x10+y10*y10);
       		double A4 = c4/(z9*z10);
       		double offset3 = Math.acos(A4)*180/PI_A3;
       		
       		sunFirst += (offset3 - offset2);
       	}
       	aveFirst = sunFirst/ arrPoint.length-3;
       	features[17] = aveFirst;
       	return new LabeledPoint(label,Vectors.dense(features));
       });
       return labeledPoint2;
	}  	

}
