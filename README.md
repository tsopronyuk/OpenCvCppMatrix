# OpenCvCppMatrix

## //helper functions

Mat importdata(string filename, char sep=',', int cols=3);

Mat repmat(const Vec<float,3>& v, int r);

Mat cross(const Mat& a, const Mat& b);

Vec<float,3> cross (const Vec<float,3>& a, const Vec<float,3>&  b);

Vec<float,3> mult(const Vec<float,3>& v, const Mat& m);

void Meshgrid(const Mat &xgv, const Mat &ygv, Mat1i &U, Mat1i &V);

void Meshgrid(const Range &xgv, const Range &ygv, Mat1i &U, Mat1i &V);

double Median(vector<float> v);




## //main functions

Mat RotationAroundGivenAxis(Vec<float,3> n,float phi);

void ReadNormalsAndDirectionsFromFile(string FileName, Mat& Normals, Mat& deltaL,Mat& Distances, Mat& Testd);

Mat GenerateOrientation(Vec<float,3> a);

int CheckSymmetryEquivalence(const Mat& UA);

Mat GenerateNet(int Nu);

Mat Angle12(const Mat& V1,const Mat& V2);

Mat ProjectABCtoPolyhedronFaces3(const Mat& UA, const Mat& Normals,const Mat& dL,const Mat&  uvw,Vec3d alpha_min_max);

Mat PredictEasyPolishingDirection(const Mat& UA,const Mat&V);

void ScanAllOrientations_FinalVersion(const Mat& Normals,const Mat& deltaL,int ScanStep, Mat& UA, Mat& Table);




## // test functions

void testRotationAroundGivenAxis();

void testReadNormalsAndDirectionsFromFile();

void testGenerateOrientation();

void testCheckSymmetryEquivalence();

void testMeshgrid();

void testGenerateNet();

void testAngle12();

void testProjectABCtoPolyhedronFaces3();

void testPredictEasyPolishingDirection();

void testScanAllOrientations_FinalVersion();
