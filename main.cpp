//
//  main.cpp
//  OpenCvMatrix
//
/*
 The list of function from matlab:
     Helper functions
 
     RotationAroundGivenAxis()
     ReadNormalsAndDirectionsFromFile()
     GenerateOrientation()
     CheckSymmetryEquivalence()
     GenerateNet()
     ScanAllOrientations_FinalVersion()
     Angle12()
     PredictEasyPolishingDirection()
 
    //ADDITIONAL WORK
     ProjectABCtoPolyhedronFaces3()
 */


//#include"stdafx.h"
#include<fstream>
#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>



using namespace cv;
using namespace std;

//helper functions
Mat importdata(string filename, char sep=',', int cols=3);
Mat repmat(const Vec<float,3>& v, int r);
Mat cross(const Mat& a, const Mat& b);
Vec<float,3> cross (const Vec<float,3>& a, const Vec<float,3>&  b);
Vec<float,3> mult(const Vec<float,3>& v, const Mat& m);
void Meshgrid(const Mat &xgv, const Mat &ygv, Mat1i &U, Mat1i &V);
void Meshgrid(const Range &xgv, const Range &ygv, Mat1i &U, Mat1i &V);
double Median(vector<float> v);

//main functions
Mat RotationAroundGivenAxis(Vec<float,3> n,float phi);
void ReadNormalsAndDirectionsFromFile(string FileName, Mat& Normals, Mat& deltaL,Mat& Distances, Mat& Testd);
Mat GenerateOrientation(Vec<float,3> a);
int CheckSymmetryEquivalence(const Mat& UA);
Mat GenerateNet(int Nu);
Mat Angle12(const Mat& V1,const Mat& V2);
Mat ProjectABCtoPolyhedronFaces3(const Mat& UA, const Mat& Normals,const Mat& dL,const Mat&  uvw,Vec3d alpha_min_max);
Mat PredictEasyPolishingDirection(const Mat& UA,const Mat&V);
void ScanAllOrientations_FinalVersion(const Mat& Normals,const Mat& deltaL,int ScanStep, Mat& UA, Mat& Table);

// test functions
void testRotationAroundGivenAxis();
void testReadNormalsAndDirectionsFromFile();
void testGenerateOrientation();
void testCheckSymmetryEquivalence();
void testMeshgrid();
void testGenerateNet();
void testAngle12();
void testProjectABCtoPolyhedronFaces3();
void testPredictEasyPolishingDirection();
//void testScanAllOrientations_FinalVersion();

int main()
{
    //run tests
    testRotationAroundGivenAxis();
    testReadNormalsAndDirectionsFromFile();
    testGenerateOrientation();
    testCheckSymmetryEquivalence();
    testMeshgrid();
    testGenerateNet();
    testAngle12();
    testProjectABCtoPolyhedronFaces3();
    testPredictEasyPolishingDirection();
    //testScanAllOrientations_FinalVersion();
    
    //main part
    cout<<"\n///////////////   MAIN PART       /////////////////\n";
    Mat Normals;
    Mat deltaL;
    Mat Distances;
    Mat Testd;
    Mat UA;
    Mat Table;
    
    string FileName="Stone13-Stage1-Data.txt";
    ReadNormalsAndDirectionsFromFile(FileName, Normals, deltaL,Distances,Testd);
    
    cout<<"\nNormals="<<Normals<<endl;
    cout<<"\ndeltaL="<<deltaL<<endl;
    cout<<"\nDistances="<<Distances<<endl;
    
    
    Mat uvw=importdata("uvw_100_your.txt");
    int ScanStep=6;
    ScanAllOrientations_FinalVersion(Normals,deltaL,ScanStep,UA,Table);
    cout<<"\nUA="<<UA<<endl;
    cout<<"\nTable="<<Table<<endl;
    
    Mat V = importdata("Stone13-Stage2-Data.txt");
    Mat phi = PredictEasyPolishingDirection(UA,V);
    cout<<"\nphi="<<phi<<endl;
    
    return 0;
}


///////////////////////  helper functions ///////////////////////
Mat repmat(const Vec<float,3>& v, int r){
    cv::Mat m_expanded = cv::Mat::zeros(r, 3, CV_32F);
    for (int row = 0; row < r; row++)
    {
        for (int col = 0; col < 3; col++)
            m_expanded.at<float>(row,col) = v(col);
        
    }
    return m_expanded;
}
Vec<float,3> cross (const Vec<float,3>& a, const Vec<float,3>&  b) // cross
{
    Vec<float,3> res;
    res(0)=a(1)*b(2)-a(2)*b(1);
    res(1)=a(2)*b(0)-a(0)*b(2);
    res(2)=a(0)*b(1)-a(1)*b(0);
    
    return res;
}

Mat cross(const Mat& a, const Mat& b){
    Mat res=a.clone();
  
    for (int i = 0; i<a.rows; i++)
        for (int j = 0; j<a.cols; j++)
            res.at<float>(i,j)=a.at<float>(i,j)*b.at<float>(i,j);
    return res.clone();
}
Vec<float,3> mult(const Vec<float,3>& v, const Mat& m){
    Vec<float,3> res={0,0,0};
    for (int i = 0; i<m.rows; i++)
        for (int j = 0; j<3; j++)
            res(i)+=v(j)*m.at<float>(i,j);
    return res;
}


void Meshgrid(const Mat &xgv, const Mat &ygv, Mat1i &X, Mat1i &Y)
{
    repeat(xgv.reshape(1,1), int(ygv.total()), 1, X);
    repeat(ygv.reshape(1,1).t(), 1, int(xgv.total()), Y);
}


void Meshgrid(const Range &xgv, const Range &ygv, Mat1i &X, Mat1i &Y)
{
    std::vector<int> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++)
        t_x.push_back(i);
    for (int i = ygv.start; i <= ygv.end; i++)
        t_y.push_back(i);
    Meshgrid(Mat(t_x), Mat(t_y), X, Y);
}

double Median(vector<float>  v)
{
    size_t size = v.size();
    if (size == 0)
    {
        return 0;  // Undefined
    }
    else
    {
        sort(v.begin(), v.end());
        if (size % 2 == 0)
        {
            return (v[size / 2 - 1] + v[size / 2]) / 2;
        }
        return v[size / 2];
    }
}


//To read data from a text file.
//filename is the name of the text file
//rows and cols show dimensions of the matrix written in the text file
Mat importdata(string filename, char sep, int cols)
{
    float x;
    int rows=0;
    ifstream in(filename);
    string line;
    while (getline(in, line))
    {
        if(!line.empty() && line[0]!='\r')
            rows++;
    }
    in.close();
    
   
    ifstream fileStream(filename);
    Mat out = Mat::zeros(rows, cols, CV_32F);//Matrix to store values
    
    
    int cnt = 0;//index starts from 0
    while ( getline(fileStream, line))
    {
        if(!line.empty() && line[0]!='\r'){
        
            istringstream iss(line);
            
            string token;
            while (getline(iss, token, sep)) { //read one token, delimited by sep
                if(token[0] =='\r')
                    break;
                
                x=stof(token);
                int temprow = cnt / cols;
                int tempcol = cnt % cols;
                out.at<float>(temprow, tempcol) = x;
                cnt++;
            }
        }
    }
    fileStream.close();
    return out;
}



//////////////////////////// main functions //////////////////////////////////
Mat ProjectABCtoPolyhedronFaces3(const Mat& UA, const Mat& Normals,const Mat& dL,const
                                 Mat&  uvw,Vec3d alpha_min_max){
    //UA is the orinetation matrix (its columns are the vectors a,b,c)
    //n is the matrix Nx3 where N is the number of facets.
    //L is the easy polishing direction in each face
    int NFaces = Normals.rows;
    Mat T = 1800 * Mat::ones(NFaces,2, CV_32F);
    Mat a=Mat::ones(uvw.rows,UA.rows, CV_32F);
    a = uvw*UA.t(); //This calculates the Cartesian coordinates if the uvw directions NB, it is assumed that the lengths of a-vectors are "1"
    //cout<<"uvw="<<uvw<<endl;
    //cout<<"UA="<<UA<<endl;
    //cout<<"a="<<a<<endl;
    int Ndirs = uvw.rows;
    int m=0;
    int exitflag = 0;
    while ((m < NFaces)&&(exitflag == 0)){
        
        Mat n = repeat(Mat(Normals.row(m).clone()),Ndirs,1);
        //cout<<"n="<<n<<endl;
        
        Mat axn=cross(a,n).clone();
        //cout<<"axn="<<axn<<endl;
        
        //rows sum
        Mat rowSum;
        for (int i = 0; i<axn.rows; i++)
            rowSum.push_back(float(cv::sum(axn.row(i))[0]));
        
        
        Mat an =  repeat(rowSum,1,3).clone();
        //cout<<"an="<<an<<endl;
        //cout<<"dL="<<dL<<endl;
        //cout<<"dL.row(m)="<<dL.row(m).clone()<<endl;
        Mat Ref = repeat(Mat(dL.row(m).clone()),Ndirs,1).clone();
        //cout<<"Ref="<<Ref<<endl;
        
        Mat nxan=cross(n,an).clone();
        //cout<<"nxan="<<nxan<<endl;
        
        Mat Pa = a-nxan; //This function calculates the projection of a to the plane perpendicular to n
        //cout<<"Pa="<<Pa<<endl;
        Mat alpha_a = (-1* Angle12(a,n)+90); //This calculates the angle between a and the plane
        
        //cout<<"alpha_a="<<alpha_a<<endl;
        //cout<<"Ref="<<Ref<<endl;
        Mat phis = Angle12(Pa,Ref).clone(); //This calculates the angle between Pa and the easy polishing directions
        //cout<<"phis="<<phis<<endl;
        //Searches thos where the angle between a and the plane is between two given values
        vector<float> partPhis;
        vector<float> index;
        int pos=0;
        for (int i=0;i<alpha_a.rows;i++){
            for (int j=0;j<alpha_a.cols;j++){
                //index = find((alpha_a > alpha_min_max(1))&(alpha_a < alpha_min_max(2))&(phis<alpha_min_max(3))); %Searches thos where the angle between a and the plane is between two given values
                
                if((alpha_a.at<float>(i,j) > alpha_min_max(0)) &
                   (alpha_a.at<float>(i,j) < alpha_min_max(1)) &
                   (phis.at<float>(i,j) < alpha_min_max(2))){
                    
                    index.push_back(pos);
                    partPhis.push_back(float(phis.at<float>(i,j)));
                    
                }
                
                pos++;
            }
        }
        
        if (index.size()==0){
            T=Mat::zeros(0,0, CV_32F);
            exitflag = 1; //In principle we do not need to continue
        }else{
            auto minIt = std::min_element(partPhis.begin(), partPhis.end());
            //type of minIt will be inferred by the compiler itself
            
            float minElement = *minIt;
            int minInd = minIt - partPhis.begin();
            
            
            int ind = index[minInd];
            //cout<<"alpha_a="<<alpha_a<<endl;
            //cout<<"phis="<<phis<<endl;
            
            T.at<float>(m,0)=alpha_a.at<float>(ind,0);
            T.at<float>(m,1)=phis.at<float>(ind,0);
            //cout<<"T="<<T<<endl;
            
        }
        m = m + 1;
    }
    //cout<<"T="<<T<<endl;
    return T;
}

//This function predicts the easy polishing direction on the face V. phi is
//the angle between the direction of easy polishing and V1 --> V2
//The format of the data V is the 3x3 matrix. Each row of the matrix is the
//Cartesian coordinate of the point on the plane V1, V2, V3
Mat PredictEasyPolishingDirection(const Mat& UA,const Mat&V){
    Mat uvw=importdata("uvw_100.txt");//{{1,0,0}, {0,1,0}, {0,0,1}, {-1,0,0}, {0,-1,0}, {0,0,-1}};
    int Ndirs = uvw.rows;
    cout<<"UA="<<UA<<endl;
     cout<<"V="<<V<<endl;
    Mat a=Mat::ones(uvw.rows,UA.rows, CV_32F);
    a = uvw*UA.t(); //This calculates the Cartesian coordinates if the uvw directions
    cout<<"a="<<a<<endl;
    
    // We find the vector connecting V1->V2
    Mat M12 =V.row(1)-V.row(0);
    Vec<float,3> V12(M12);
    cout<<"V12="<<V12<<endl;
    
    //We find the vector connecting V1->V3
    Mat M13 =V.row(2)-V.row(0);
    Vec<float,3> V13(M13);
    cout<<"V13="<<V13<<endl;
    
    //cross product is the normal to the V1, V2, V3 plane
    Vec<float,3> n=V12.Vec<float,3>::cross(V13);
    cout<<"n="<<n<<endl;
    
    //normalize the vector n, to make sure the length of it is equal to 1
    n = n/norm(n);
    cout<<"n="<<n<<endl;
    cout<<"V="<<V<<endl;
    Vec<float,3> prod=mult(n,V);
    cout<<prod<<endl;
    
cv:Scalar tempVal = mean(prod); //The dot products between the vector n and the vector V1,V2,V3
    float d = tempVal.val[0];
    //cout<<"d = "<< d <<endl;
    if (d<0)
        n = -n; //ust to make sure that the vector n looks outside the polyhedron crystal and not inside
    
    
    Mat n2 = repmat(n,Ndirs);
    cout<<"n2="<<n2<<endl;
    
    Mat axn2=cross(a,n2).clone();
    cout<<"axn2="<<axn2<<endl;
    
    //rows sum
    Mat rowSum;
    for (int i = 0; i<axn2.rows; i++)
        rowSum.push_back(float(cv::sum(axn2.row(i))[0]));
    
    Mat an =  repeat(rowSum,1,3).clone();
    
    Mat n2xan=cross(n2,an).clone();
    cout<<"n2xan="<<n2xan<<endl;
    
    Mat Pa = a-n2xan; //This function calculates the projection of a to the plane perpendicular to n2
    cout<<"Pa="<<Pa<<endl;
    Mat alpha_a = (-1* Angle12(a,n2)+90); //This calculates the angle between a and the plane
    cout<<"alpha_a="<<alpha_a<<endl;
    
    //Searches thos where the angle between a and the plane is between two given values
    vector<float> partAlpha;
    vector<float> index;
    int pos=0;
    for (int i=0;i<alpha_a.rows;i++){
        for (int j=0;j<alpha_a.cols;j++){
            //index = find((alpha_a > alpha_min_max(1))&(alpha_a < alpha_min_max(2))&(phis<alpha_min_max(3))); %Searches thos where the angle between a and the plane is between two given values
            
            if(alpha_a.at<float>(i,j)<1){
                
                index.push_back(pos);
                partAlpha.push_back(float(fabs(alpha_a.at<float>(i,j))));
                
            }
            
            pos++;
        }
    }
    cout<<"pos="<<pos<<endl;
    Mat phi=Mat::zeros(0, 0, CV_32F);
    if(partAlpha.size()){
        auto minIt = std::min_element(partAlpha.begin(), partAlpha.end());
        //type of minIt will be inferred by the compiler itself
        
        float minElement = *minIt;
        int minInd = minIt - partAlpha.begin();
        int ind2 = index[minInd];
        
        cout<<"ind2="<<ind2<<endl;
        phi = Angle12(Pa.row(ind2),Mat(V12)); //The angle between the direction V12 and the easy polishing one
        cout<<"phi="<<phi<<endl;
        Vec<float,3> Pa2=Pa.row(ind2);
        
        Mat M=Mat::zeros(3, 3, CV_32F);
        Mat(V12).col(0).copyTo(M.col(0));
        Mat(Pa2).col(0).copyTo(M.col(1));
        Mat(-n).col(0).copyTo(M.col(2));
        cout<<"M="<<M.t()<<endl;
        //Mat M = [V12; Pa(ind2,:); -n];
        if (determinant(M.t()) <0)
            phi = -phi;
        //cout<<"determinant="<<determinant(M.t())<<endl;
    }
    return phi;
}



//This function calculates the angle between vectors V1 and V2
Mat Angle12(const Mat& V1_,const Mat& V2_){
    cout << "V1_=" << V1_ << endl;
    cout << "V2_=" << V2_<< endl;
    
    /*
     function A = Angle12(V1,V2)
     %This function calculates the angle between vectors V1 and V2
     % V1 and V2 are matrices Nx3, each row of these matrices are Cartesian
     % coordinates of the vector
     
     L1 = sum(V1.^2,2).^(-0.5); %1/length
     L2 = sum(V2.^2,2).^(-0.5); %1/length
     
     A = acosd(sum(V1.*V2,2).*L1.*L2);
     */
    Mat V1=V1_;
    if(V1_.cols !=3)
        V1=V1.t();
    
    Mat V2=V2_;
    if(V2_.cols !=3)
        V2=V2.t();
    
    
    cout << "V1=" << V1 << endl;
    cout << "V2=" << V2 << endl;
    
    Mat A =  Mat::zeros(V1.rows, 1, CV_32F);
    
    for (int i=0;i<V1.rows;i++){
        //L1 = sum(V1.^2,2).^(-0.5);
        float l1=1/sqrt(V1.at<float>(i,0)*V1.at<float>(i,0)+
                      V1.at<float>(i,1)*V1.at<float>(i,1)+
                      V1.at<float>(i,2)*V1.at<float>(i,2));
        
        
        //L2 = sum(V2.^2,2).^(-0.5);
        float l2=1/sqrt(V2.at<float>(i,0)*V2.at<float>(i,0)+
                      V2.at<float>(i,1)*V2.at<float>(i,1)+
                      V2.at<float>(i,2)*V2.at<float>(i,2));
        
        //sum(V1.*V2,2)
        float scal= V1.at<float>(i,0)*V2.at<float>(i,0)+
                    V1.at<float>(i,1)*V2.at<float>(i,1)+
                    V1.at<float>(i,2)*V2.at<float>(i,2);
        
        //A = acosd(sum(V1.*V2,2).*L1.*L2);
        A.at<float>(i,0)=float(acos(scal*l1*l2)*180/M_PI);
        
    }
    
    return A;
}

//This function generates a uniform net on the sphere
Mat GenerateNet(int Nu){
    cv::Mat1i U, V;

    Meshgrid(cv::Range(-2*Nu,2*Nu), cv::Range(-2*Nu,2*Nu), U, V);
    
    Mat L =  Mat::zeros(U.rows, U.cols, CV_32F);
    Mat index0=U.clone();
    Mat index=U.clone();
    Mat RHO=L.clone();
    Mat X=L.clone();
    Mat Y=L.clone();
    Mat Z=L.clone();
    Mat Xs=L.clone();
    Mat Ys=L.clone();
    
    //L = (U.^2 + V.^2 - U.*V).^0.5
    for (int i=0;i<U.rows;i++){
        for (int j=0;j<U.cols;j++){
            L.at<float>(i,j)=sqrt(U.at<int>(i,j)*U.at<int>(i,j)+
                              V.at<int>(i,j)*V.at<int>(i,j)-
                                  U.at<int>(i,j)*V.at<int>(i,j));
            //index0 = (L == 0)
            index0.at<int>(i,j) = L.at<float>(i,j)==0;
            //RHO = pi/2/Nu*L;
            float rho = M_PI/2/Nu*L.at<float>(i,j);
            RHO.at<float>(i,j) =rho;
            
            //X = sin(RHO).*(U - V/2)./L;
            X.at<float>(i,j) = sin(rho)*(U.at<int>(i,j) - V.at<int>(i,j)/2.0)/L.at<float>(i,j);
            //Y = sin(RHO).*V./L*sqrt(3)/2;
            Y.at<float>(i,j) = sin(rho)*V.at<int>(i,j)/L.at<float>(i,j)*sqrt(3.0)/2.0;
            //Z = cos(RHO);
            Z.at<float>(i,j) = cos(rho);
            
            //X(index0) = 0; Y(index0) = 0; Z(index0) = 1;
            if (index0.at<int>(i,j)){
                X.at<float>(i,j) = 0;
                Y.at<float>(i,j) = 0;
                Z.at<float>(i,j) = 1;
                
            }
            
            //Xs = X./(1+Z);
            //Xs.at<float>(i,j) = X.at<float>(i,j)/(1+Z.at<float>(i,j));
            //Ys = Y./(1+Z);
            //Ys.at<float>(i,j) = Y.at<float>(i,j)/(1+Z.at<float>(i,j));
            
            //index = (RHO<(pi/2));
            index.at<int>(i,j) = rho < M_PI/2;
        }
    }
    
    //cout<<"X="<<X<<endl;
    //cout<<"Y="<<Y<<endl;
    //cout<<"Z="<<Y<<endl;
    
    //XYZ = [X(index),Y(index),Z(index)];
    vector<float> x,y,z;
    for (int j=0;j<U.cols;j++){
        for (int i=0;i<U.rows;i++){
    
            if(index.at<int>(i,j)){
                x.push_back(X.at<float>(i,j));
                y.push_back(Y.at<float>(i,j));
                z.push_back(Z.at<float>(i,j));
            }
        }
    }
    
    Mat XYZ=Mat::zeros(int(x.size()), 3, CV_32F);
    //cout<<"size="<<x.size()<<endl;
    for (int i=0;i<x.size();i++){
        XYZ.at<float>(i,0)=x[i];
        XYZ.at<float>(i,1)=y[i];
        XYZ.at<float>(i,2)=z[i];
    }

    return XYZ;
    
}

//This function will generate the vectors a,b,c from the direction of a
Mat GenerateOrientation(Vec<float,3> a){
    a = a / norm(a); //Make sure that the length of the vector a is equal to "1"
        Vec<float,3> V = {0,1,0},V1 = {0,0,1};
    Vec<float,3> b = a.cross(V); //cross(a,V); //Finds the vector b, which is perpendicular to a
    if (norm(b) < 1e-5)
        b = V1; //This is the case when a = [0,1,0] and then the cross product has the length zero, in this case we can choose vector b manually
    else
        b = b / norm(b); //Otherwise just normalize vector b

    Vec<float,3> c = a.cross(b);  //cross(a,b); //Now c is easy to calculate from the cross product between a and b

        //We find the matrix, whose first column is "a", second column is "b" and third column is "c"
    Mat UA=Mat::zeros(3, 3, CV_32F);
    Mat(a).col(0).copyTo(UA.col(0));
    Mat(b).col(0).copyTo(UA.col(1));
    Mat(c).col(0).copyTo(UA.col(2));
    return UA;
}

int CheckSymmetryEquivalence(const Mat& UA){
    Mat inverse=UA.inv();
    //cout<<"inverse="<<inverse<<endl;
    /*
    Mat M=Mat::zeros(3, 1, CV_32F);
    M.at<float>(0,0)=0;
    M.at<float>(1,0)=0;
    M.at<float>(2,0)=1;
     */
    Vec3d v={0,0,1};
    
    //Mat m = inverse*V;
    Vec3d C = mult(v,inverse);
    
    
    //Vec3d C=m.col(0);
    //cout<<"C="<<C<<endl;
    //V = (C(1)>=0)&(C(2)>=0)&(C(3)>=0)&(C(1)>=C(3))&(C(2)>=C(3));
    int V=(C(0)>=0)&&(C(1)>=0)&&(C(2)>=0)&&(C(0)>=C(2))&&(C(1)>=C(2));
    if(V)
        return 1;
    return 0;
}

void ReadNormalsAndDirectionsFromFile(string FileName, Mat& Normals, Mat& deltaL,Mat& Distances, Mat& Testd){
    /*
     function [Normals,deltaL,Distances,Testd] = ReadNormalsAndDirectionsFromFile(FileName)
     %This function reads the file FileName, calculates Normals to the plane,
     %the direction of easy polishing. In addition it does the following
     
     % a) checks if both L points are in the plane.
     % d) Calculates the difference vectors between L1 and L2
     
     %To check if the L points are on the plane we will calculate dL - this the
     %the distance between them and the V1V2V3 plane. These numbers must be zero
     %or very close to zero
     
     VL = importdata(FileName); %Reads the table from the file. This table should have the size of 6Nx3 where N is the number of measured faces
     NFaces = round(size(VL,1)/6); %here we calculate the number of the faces. round is used here just to make sure that the number is integer.
     
     %I think the next block will look differently in C++. In MatLab it just
     %creates the inital arras if requred size
     %--------------------------------------------------------
     Normals = zeros(NFaces,3);
     deltaL = Normals;
     Distances = zeros(NFaces,1);
     Testd = zeros(NFaces,2);
     %---------------------------------------------------------
     
     for m = 1:NFaces
     N = 6*(m-1) + [1,3,4,5,6];
     V = VL(N(1):N(2),:); %We read the coordinates of three vertices of the face m.
     L = VL(N(3):N(4),:); %We read the coordinates of two points L1 and L2 of the face m,
     V12 = V(2,:)-V(1,:); %We find the vector connecting V1->V2
     V13 = V(3,:)-V(1,:); %We find the vector connecting V1->V3
     
     n = cross(V12,V13); %cross product is the normal to the V1, V2, V3 plane
     n = n/norm(n); %normalize the vector n, to make sure the length of it is equal to 1
     n3 = repmat(n,3,1); %creates the matrix with three rows, each is vector n
     
     dV = sum(V.*n3,2); %Calculates the dot products V1*n, V2*n, V3*n. All these numbers will be same (it is the distance to the plane) and should be positive
     if prod(dV)<0
     %If not, invert the vector n
     n = -n;
     dV = -dV;
     end
     %The procedure above will make sure that the vector n points outside the
     %crystal and not inside the crystal
     
     d0 = mean(dV);%This the distance to the plane, mean is not really necessary here because dV are the same
     n2 = repmat(n,2,1);  %creates the matrix with two rows, each is vector n
     dL = sum(L.*n2,2); %Calculates the dot products L1*n, L2*n. If L1L2 points are in the plane then it must be equal to d0
     
     dL = dL - d0; %Check this
     
     Normals(m,:) = n;
     deltaL(m,:) = L(2,:) - L(1,:); %This is the vector which points from L1 to ---> L2
     Distances(m) = d0;
     Testd(m,:) = dL;  %Close to zero values of Testd must indicate that the data are measured correctly and that the points L1, L2 are on the V1,V2,V3 planes
     
     %Now our task is to rotate the direction L1 ---> L2 by the given angle around the plane normal.
     
     %Let us read the angle from the data
     Phi = VL(N(5),1);
     L12before = deltaL(m,:);
     
     % Calls another function to claculate the rotation matrix
     M = RotationAroundGivenAxis(-n,Phi);  %here we use -n because the rotation is anticlockwise when we
     L12after = (M*L12before')';  %Multiplication of the matrix M with the bevtor L12before gives the vector L12after, rotated by the angle phi
     deltaL(m,:) = L12after;
     end
    
     */
    
    Mat VL = importdata(FileName); //Reads the table from the file. This table should have the size of 6Nx3 where N is the number of measured faces
    
    //cout<<"VL="<<VL<<endl;
    int NFaces = round(VL.rows/6); //here we calculate the number of the faces. round is used here just to make sure that the number is integer.
    
    //Matrix to store values zeros(NFaces,3);
    Mat _Normals = Mat::zeros(NFaces, 3, CV_32F);
    Mat _deltaL = Mat::zeros(NFaces, 3, CV_32F);;
    Mat _Distances = Mat::zeros(NFaces, 1, CV_32F);
    Mat _Testd = Mat::zeros(NFaces, 2, CV_32F);
    
    Normals = _Normals.clone();
    deltaL = _deltaL.clone();
    Distances = _Distances.clone();
    Testd =_Testd.clone();
    
    vector<int> initN={0,1,3,4,5,6};
   
    for(int m = 1; m<=NFaces; m++){
        
        vector<int> N=initN;
        int shift= 6*(m-1);
        
        //We read the coordinates of three vertices of the face m.
        Mat V= Mat::zeros(3, 3, CV_32F);
        
        int nrow=0;
        for (int i=N[1]+shift;i<=N[2]+shift;i++){
            //cout<<VL.row(i-1)<<endl;
            
            for (int j=0;j<3;j++)
                V.at<float>(nrow,j)=VL.at<float>(i-1,j);
            nrow++;
        }
        //cout<<"V="<<V<<endl;
       
        //We read the coordinates of two points L1 and L2 of the face m
       Mat L= Mat::zeros(2, 3, CV_32F);
        nrow=0;
        for (int i=N[3]+shift;i<=N[4]+shift;i++){
            for (int j=0;j<3;j++)
                L.at<float>(nrow,j)=VL.at<float>(i-1,j);
            nrow++;
        }
      
       //cout<<"L="<<L<<endl;
        
        // We find the vector connecting V1->V2
        Mat M12 =V.row(1)-V.row(0);
        Vec<float,3> V12(M12);
        //cout<<"V12="<<V12<<endl;
        
        //We find the vector connecting V1->V3
        Mat M13 =V.row(2)-V.row(0);
        Vec<float,3> V13(M13);
        //cout<<"V13="<<V13<<endl;
        
        //cross product is the normal to the V1, V2, V3 plane
        Vec<float,3> n=V12.Vec<float,3>::cross(V13);//cross(V12,V13);
    
        //cout<<"n="<<n<<endl;
        //normalize the vector n, to make sure the length of it is equal to 1
        n = n/norm(n);
        //cout<<"n="<<n<<endl;
        
        //creates the matrix with three rows, each is vector n
        Mat n3 = repmat(n, 3);
        //cout<<"n3="<<n3<<endl;
       
        Mat originalMatrix=V.mul(n3);
       
        Mat dV;
        float prod=1;
        //rows sum
        for (int i = 0; i<originalMatrix.rows; i++){
            float item=cv::sum(originalMatrix.row(i))[0];
            dV.push_back(float(item));
            prod*=item;
        }
        //cout<<"dv="<<dV<<endl;
        //cout<<"prod="<<prod<<endl;
        if (prod<0){
            //If not, invert the vector n
            n = -n;
            dV = -dV;
        }
        
    //This the distance to the plane, mean is not really necessary here because dV are the same
    cv:Scalar tempVal = mean(dV);
    float d0 = tempVal.val[0];
    //cout<<"d0 = "<< d0 <<endl;
    
    Mat n2 = repmat(n,2);  //creates the matrix with two rows, each is vector n
    
    //Calculates the dot products L1*n, L2*n. If L1L2 points are in the plane then it must be equal to d0
    Mat dL;
    originalMatrix=L.mul(n2);
    for (int i = 0; i<originalMatrix.rows; i++){
            float item=cv::sum(originalMatrix.row(i))[0];
            dL.push_back(float(item));
        }
        
    dL = dL - d0; //Check this
    
    Mat Mn(n);
    //cout<<"Mn="<<Mn<<endl;
    Mn=Mn.t();
    Mn.row(0).copyTo(Normals.row(m-1));
   
    //cout<<"Normals="<<Normals<<endl;
    
    //This is the vector which points from L1 to ---> L2
    //Vec3d diff=L.row(1) - L.row(0);
    deltaL.row(m-1) = (L.row(1) - L.row(0)+0);
    //cout<<"L="<<L<<endl;
    //cout<<"deltaL="<<deltaL<<endl;
        
    Distances.row(m-1) = d0;
    //cout<<"Distances="<<Distances<<endl;
        
    //close to zero values of Testd must indicate that the data are measured correctly and that the points L1, L2 are on the V1,V2,V3 planes
    Testd.row(m-1) = (Mat(dL).t().row(0)+0);
    //cout<<"Testd="<<Testd<<endl;
     
    
    //Now our task is to rotate the direction L1 ---> L2 by the given angle around the plane normal.
    
    //Let us read the angle from the data
      float Phi;
      Phi=VL.at<float>(N[5]+shift-1,0);
        
     //cout<<"Phi="<<Phi<<endl;
     
    Mat L12before= Mat::zeros(1, 3, CV_32F);
    nrow=0;
    
    for (int j=0;j<3;j++)
        L12before.at<float>(0,j)=deltaL.at<float>(m-1, j);
        
     //cout<<"L12before="<<L12before<<endl;
        
    //cout<<"n="<<n<<endl;
        
    // Calls another function to claculate the rotation matrix
    Mat M = RotationAroundGivenAxis(-1*n,Phi);  //here we use -n because the rotation is anticlockwise when we
    //cout<<"M="<<M<<endl;
        
    Mat L12after = (M*L12before.t()).t();  //Multiplication of the matrix M with the bevtor L12before gives the vector L12after, rotated by the angle phi
    //cout<<"L12after="<<L12after<<endl;
    
    deltaL.row(m-1) = (L12after.row(0)+0);
    //cout<<"deltaL="<<deltaL<<endl;
    }

}

Mat RotationAroundGivenAxis(Vec<float,3> n,float phi){
    /*
     function M = RotationAroundGivenAxis(n,phi)
     %This function calculates the rotation matrix around the axis, given by the
     %vector n and by the angle phi
     
     n = n / norm(n); %normalize the direction of rotation
     CV = [1,0,0];
     if (abs(n(1))>0.5)
     CV = [0,1,0];
     end;
     
     e2 = cross(n,CV); e2 = e2/norm(e2); e3 = cross(n,e2);
     cx = cosd(phi); sx = sind(phi);
     C = [n',e2',e3'];
     Mx = [1,     0,    0;
     0,    cx,  -sx;
     0,    sx,   cx];
     M = C*Mx*C';
     */
    
    //normalize the direction of rotation
    n=n/norm(n);
    Vec<float,3> CV = {1,0,0}, CV1={0,1,0};
    if (fabs(n(0))>0.5)
        CV = CV1;
   
    Vec<float,3> e2 =n.Vec<float,3>::cross(CV);
    
    
    e2 = e2/norm(e2);
    Vec<float,3> e3 = n.Vec<float,3>::cross(e2);
    
    float cx = cos(phi*M_PI / 180);
    float sx = sin(phi*M_PI / 180);
    
    //create Mat from vector
    vector<float> vec; // vector with your data
    for(int i=0;i<3;i++)
        vec.push_back(n(i));
    for(int i=0;i<3;i++)
        vec.push_back(e2(i));
    for(int i=0;i<3;i++)
        vec.push_back(e3(i));
    
    //create Mat from vector
    Mat C(3,3, CV_32F);
    memcpy(C.data, vec.data(), vec.size()*sizeof(float));
    C=C.t();
    
    //std::cout << "C = " << C <<std::endl;
    vector<float> vx={1,     0,    0,
        0,    cx,  -sx,
        0,    sx,   cx};
    
    //create Mat from vector
    Mat Mx(3,3, CV_32F);
    memcpy(Mx.data, vx.data(), vx.size()*sizeof(float));
    //std::cout << "Mx = " << Mx <<std::endl;
    
    Mat M(3,3, CV_32F);
    
    
    M = C*Mx*C.t();;
    return M;
}


//This function tests large number of crystal orientations and choose the one, which is the best.
void ScanAllOrientations_FinalVersion(const Mat& Normals,const Mat& deltaL,int ScanStep, Mat& UA, Mat& Table){
    //The ScanStep is given by the angle ScanStep (in degrees).
    //The input data is given by the Normals (NFaces x 3 matrix) and deltaL (NFaces x 1 matrix)
    //uvw are crystallographic directions, whose projections can be easy polishing ones, Choose any out of uvw_100.mat, uvw_110.mat, uvw_111.mat (load the mat files stored in the same folder).
    //In fact, I am pretty sur that we need to choose uvw_100. I will keep this
    //variavle for now but delete it in future.
    
    Vec3d PhiMax = {-55,2,45}; //this sets some limiting angles, which are used in the program later. The first and the second angle are the boundaries for the angles between "a" and the face, the third angle is the maximum difference between caclualted and observed direction
    Mat uvw=importdata("uvw_100.txt");//{{1,0,0}, {0,1,0}, {0,0,1}, {-1,0,0}, {0,-1,0}, {0,0,-1}};
    
    int N = 90/ScanStep;//Generates the mesh of directions for the vector. It is a separate function
    Mat XYZ = GenerateNet(N); //Generates the mesh of directions for the vector. It is a separate function
    
    //Phi = 0:ScanStep:360; %The angles of rotation around vector a
    int NXYZ = XYZ.rows;  //The total numbers of vectors a, generated
    
    //Table =[]; UA=[];
    Table=Mat::zeros(0,0, CV_32F);
    UA=Mat::zeros(0,0, CV_32F);
    float Pmin = 90;
    int countV=0;

    for(int n=0;n<NXYZ; n++){
        Vec3d a=XYZ.row(n).clone();
        //cout<<"a"<<a<<endl;
        
        //XYZ.row(n).copyTo(a.col(0));//Extract the direction of the vector "a"
        Mat UA_n = GenerateOrientation(a);//Generates the orientation matrix "a","b","c". The columns of this matrix are Cartesian coordinates of "a","b","c"
        int count=1;
        for(int phi=0;phi<=360;phi+=ScanStep){
            //This is the matrix of rotation around vector a
            float m[3][3] = {
                {1, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            };
            
            m[1][1]=cos(phi*M_PI / 180); m[1][2]=-sin(phi*M_PI / 180);
            m[2][1]=sin(phi*M_PI / 180); m[2][2]=cos(phi*M_PI / 180);
            
            
            //create Mat from vector
            Mat M=Mat(3,3, CV_32F,m);
            
            Mat UA_nm = UA_n*M;
            //cout<<"M="<<M<<endl;
            //cout<<"UA_nm="<<UA_nm<<endl;
            
            int V = CheckSymmetryEquivalence(UA_nm); //This little script checks if this orinetation is not symmetry equivalent to the ones which are already processed (speeds up the process).
            
            if (V){
                
                countV++;
                
                
                Mat T = ProjectABCtoPolyhedronFaces3(UA_nm,Normals,deltaL,uvw,PhiMax); //This scripts predicts easy polishing directions by projecting uvw to the face to the faces.
                if (T.total()>0){ //The result is not empty
                    
                    //Extract the direction of the vector "a" easy polishing direction
                    vector<float> vph;
                    for(int l=0;l<T.rows;l++)
                        vph.push_back(T.at<float>(l,1));
                    
                    float P =  Median(vph); //Extracts the median
                    
                    //cout<<"ph"<<ph<<" ";
                    //cout<<"P"<<P<<endl;
                    
                    if (P < Pmin){
                        Table = T.clone();
                        UA = UA_nm.clone();
                        Pmin = P;
                    }
                }
            }
            count++;
        }
    }
}

///////////////////////  tests  ////////////////////////////////////////
void testRotationAroundGivenAxis(){
    Vec3d n(-0.4605,    0.4705,   -0.7527);
    float phi=-3.0141;

    Mat m=RotationAroundGivenAxis(n,phi);
    cout<<"\nTest for RotationAroundGivenAxis():\nM="<< m<<endl;
    /*
    the answer of test
    M = [0.99891001, -0.039878216, -0.024260338;
    0.039278779, 0.99892288, -0.024703907;
    0.02521936, 0.023724103, 0.99940038]
    */
}
    
void testReadNormalsAndDirectionsFromFile(){
    Mat Normals;
    Mat deltaL;
    Mat Distances;
    Mat Testd;
    ReadNormalsAndDirectionsFromFile("Data.txt", Normals, deltaL,Distances,Testd);
    cout<<"\nTest for ReadNormalsAndDirectionsFromFile():\n";
    cout<<"\nNormals="<<Normals<<endl;
    cout<<"\ndeltaL="<<deltaL<<endl;
    cout<<"\nDistances="<<Distances<<endl;
    cout<<"\nTestd="<<Testd<<endl;
    /*
     the answer of test
     
     
     Normals=[0.46046603, -0.47052163, 0.75271541;
     -0.48381317, -0.43415922, 0.75988853;
     -0.65996903, -0.0028302418, 0.75128746;
     0.57515419, 0.79837298, 0.17832063;
     0.52750528, 0.16702157, 0.83297175]
     
     deltaL=[2.9919925, 3.1424439, 0.13401788;
     1.348405, 0.578821, 1.7193675;
     -6.1770816, -1.18028, -3.0741835;
     -4.8644977, -5.4849205, 6.6522331;
     4.5653458, -0.082584001, -4.1594067]
     
     Distances=[1.8398435;
     2.0820284;
     1.801188;
     0.81910741;
     1.1683799]
     
     Testd=[-1.1920929e-07, 2.3841858e-07;
     2.3841858e-07, 0.40285134;
     -1.8784835, -0.10805619;
     3.7708721, -2.2197459;
     1.6327368, 0.56251967]
     */
}

void testGenerateOrientation(){
    Vec3d a(-0.7701,   -0.6379,    0.0070);
    cout<<"\nTest for GenerateOrientation():\n";
    Mat UA=GenerateOrientation(a);
    cout<<"\nUA="<<UA<<endl;
    /*
     the answer of test
     UA=[-0.77009255, -0.0090893535, 0.63786745;
     -0.6378938, 0, -0.77012432;
     0.0069999322, -0.99995869, -0.0057980423]
     */
}

void testCheckSymmetryEquivalence(){
    Vec3d a(-0.7701,   -0.6379,    0.0070);
    cout<<"\nTest for GenerateOrientation():\n";
    Mat UA=GenerateOrientation(a);
    if(CheckSymmetryEquivalence(UA))
        cout<<"\nCheckSymmetryEquivalence is 1"<<endl;
    else
        cout<<"\nCheckSymmetryEquivalence is 0"<<endl;
    /*
     the answer of test
     CheckSymmetryEquivalence is 0
     */
}

void testGenerateNet(){
    int ScanStep=6;
    int N = 90/ScanStep;
    Mat XYZ=GenerateNet(N);
    cout<<"\nTest for GenerateNet():\n";
    cout<<"\nNumber of rows="<<XYZ.rows<<endl;
    cout<<"\nXYZ="<<XYZ<<endl;
    /*
     the answer of test
     Number of rows=817
     
     XYZ=[-0.77007806, -0.6379112, 0.0069968887;
     -0.81070459, -0.58507568, 0.021083895;
     -0.84821844, -0.52889669, 0.028174324;
     -0.88214719, -0.47013038, 0.028174324;
     -0.91204268, -0.40955296, 0.021083895;
     . . .
     */
}

void testMeshgrid(){
    cv::Mat1i U, V;
    Meshgrid(cv::Range(1,3), cv::Range(10, 14), U, V);
    cout<<"\nTest for Meshgrid():\n";
    cout<<"\nU="<<U<<endl;
    cout<<"\nV="<<V<<endl;
    /*
    the answer of test
     U=[1, 2, 3;
     1, 2, 3;
     1, 2, 3;
     1, 2, 3;
     1, 2, 3]
     
     V=[10, 10, 10;
     11, 11, 11;
     12, 12, 12;
     13, 13, 13;
     14, 14, 14]
     */
    int Nu=90/6;
    Meshgrid(cv::Range(-2*Nu,2*Nu), cv::Range(-2*Nu,2*Nu), U, V);
    //cout<<"\nTest2 for Meshgrid():\n";
    //cout<<"\nU="<<U<<endl;
    //cout<<"\nV="<<V<<endl;
    
}

void testAngle12(){
    float v1[2][3] = {
        {-0.7701,   -0.6379,    0.0070},
        {0.0091,         0,    1.0000}
    };
    float v2[2][3] = {
        {0.4605,   -0.4705,    0.7527},
        {0.4605,   -0.4705,    0.7527}
    };
    
    Mat V1 = cv::Mat(2, 3, CV_32F, v1);
    Mat V2 = cv::Mat(2, 3, CV_32F, v2);
    
    Mat A=Angle12(V1,V2);
    
    cout<<"\nTest for Angle12():\n";
    cout<<"\nA="<<A<<endl;
    /*
     the answer of test
     A=[92.822144;
     40.8106]
     */
}
    
void testScanAllOrientations_FinalVersion(){
    Mat Table;
    Mat UA;
    Mat Normals;
    Mat deltaL;
    Mat Distances;
    Mat Testd;
    int ScanStep=6;
    ReadNormalsAndDirectionsFromFile("Data.txt", Normals, deltaL,Distances,Testd);
    //cout<<"\ndeltaL="<<deltaL<<endl;
    ScanAllOrientations_FinalVersion(Normals, deltaL, ScanStep, UA, Table);
    cout<<"\nTest for ScanAllOrientations_FinalVersion():\n";
    cout<<"\nUA="<<UA<<endl;
    cout<<"\nTable="<<Table<<endl;
    /*
     the answer of test
     
     UA=[0.726143, -0.6441015, 0.24051943;
     0.31442913, 0, -0.94928098;
     0.61143327, 0.76494002, 0.20252427]
     
     Table=[-45.222824, 42.998085;
     -1.3298874, 10.527698;
     1.1895218, 16.809319;
     -13.535805, 44.881142;
     -17.301895, 10.195665]
     
     
     */
    
}
void testProjectABCtoPolyhedronFaces3(){
    Mat Normals;
    Mat deltaL;
    Mat Distances;
    Mat Testd;
   
    
    ReadNormalsAndDirectionsFromFile("Data.txt", Normals, deltaL,Distances,Testd);
    //ScanAllOrientations_FinalVersion(Normals, deltaL, ScanStep, UA, Table);
    Vec3d PhiMax = {-55,2,45}; //this sets some limiting angles, which are used in the program later. The first and the second angle are the boundaries for the angles between "a" and the face, the third angle is the maximum difference between caclualted and observed direction
    Mat uvw=importdata("uvw_100.txt");//{{1,0,0}, {0,1,0}, {0,0,1}, {-1,0,0}, {0,-1,0}, {0,0,-1}};
    float uaEmpty[3][3] = {
        {-0.726142982295396,    0.678067996724639,   0.113754828824905},
        {-0.314429134723803,    -0.474640474263637,    0.822101416753207},
        {0.611433306665567,    0.561195342112375 ,    0.557861182993929}
    };
    
    float uaEmpty1[3][3] = {
        {-0.7261,    0.6781,   0.1138},
        {-0.3144,    -0.4746,    0.8221},
        {0.6114,    0.5612 ,    0.5579}
    };
    

    
    float ua[3][3] = {
        {0.7261,    -0.6441,   0.2405},
        {0.3144,    0,    -0.9492},
        {0.6114,    0.76494,    0.2025}
    };
    /* Correct T from matlab for ua
     -45.2228   42.0222
     -1.3299   10.3328
     1.1895   18.2059
     -13.5358   42.5719
     -17.3019   10.7055
     */
    float ua1[3][3] = {
        {-0.3195,    0.7470,   -0.5830},
        {-0.7116,    0.2171,    0.6682},
        {0.6257,    0.6284,    0.4622}
    };
    /* Correct T from matlab for ua1
     -41.2003   18.8674
     1.2520   12.0579
     1.2321   15.5072
     -39.8159   42.4092
     -10.8983   41.2527
     */
    
    float dl[5][3] = {
        {2.9502,    3.1793,    0.1826},
        {1.3722,    0.5409,    1.7129},
        {-6.2125,   -0.8770,   -3.1042},
        {-5.3852,   -5.0902,    6.5643},
        {4.5824,   -0.2702,   -4.1326}
    };
    
   
    
    
    //Mat UA = cv::Mat(3, 3, CV_32F, ua);
    Mat dL = cv::Mat(5, 3, CV_32F, dl);
    Mat UA = cv::Mat(3, 3, CV_32F, ua);
    Mat T = ProjectABCtoPolyhedronFaces3(UA,Normals,dL,uvw,PhiMax);
    cout<<"\nTest 1 for ProjectABCtoPolyhedronFaces3():\n";
    cout<<"\nT="<<T<<endl;
    
    UA = cv::Mat(3, 3, CV_32F, uaEmpty1);
    T = ProjectABCtoPolyhedronFaces3(UA,Normals,dL,uvw,PhiMax);
    cout<<"\nTest 2 for ProjectABCtoPolyhedronFaces3():\n";
    cout<<"\nT="<<T<<endl;
    
    /*
     the answer of test
     for ua
     T=[-45.222519, 42.022675;
     -1.3294983, 10.333882;
     1.1894073, 18.20536;
     -13.535782, 42.572025;
     -17.301964, 10.70552]
     
     for ua1
     T=[-41.200272, 18.86722;
     1.251976, 12.059441;
     1.2320709, 15.506928;
     -39.815918, 42.40921;
     -10.898338, 41.253075]
     
     for uaEmpty1
     T=[]
     */
    
}

void testPredictEasyPolishingDirection(){
    float ua[3][3] = {
        {0.7261,    -0.6441,   0.2405},
        {0.3144,    0,    -0.9492},
        {0.6114,    0.76494,    0.2025}
    };
    
    Mat UA = cv::Mat(3, 3, CV_32F, ua);
    //Mat V = importdata("Stage2-Data.csv");
    Mat V = importdata("Stage2-Data.txt", ' ');
    Mat phi=PredictEasyPolishingDirection(UA,V);
    cout<<"\nTest for PredictEasyPolishingDirection():\n";
    cout<<"\nphi="<<phi<<endl;
    /*
     the answer of test
     phi=[-15.915895]
     */
    
}


