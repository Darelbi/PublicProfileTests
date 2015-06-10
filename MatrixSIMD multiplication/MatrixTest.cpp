#undef NDEBUG
#include <assert.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <chrono>
#include <iostream>
#include <fstream>


#define TIMING

#ifdef TIMING
#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER  start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << \
    std::chrono::duration_cast<std::chrono::microseconds>( \
            t2-t1 \
    ).count() << " microseconds " << std::endl;
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif

typedef std::chrono::high_resolution_clock Clock;

struct Matrix{
    float m[16];
};

struct SMatrix{
    __m128 m[4];
};

void mmul(const float *a, const float *b, float *r)
{
  for (int i=0; i<16; i+=4)
    for (int j=0; j<4; j++)
      r[i+j] = b[i]*a[j] + b[i+1]*a[j+4] + b[i+2]*a[j+8] + b[i+3]*a[j+12];
}

void mmul_sse(const float * a, const float * b, float * r)
{
  volatile __m128 a_line, b_line, r_line;
  for (int i=0; i<16; i+=4) {
    // unroll the first step of the loop to avoid having to initialize r_line to zero
    a_line = _mm_load_ps(a);         // a_line = vec4(column(a, 0))
    b_line = _mm_set1_ps(b[i]);      // b_line = vec4(b[i][0])
    r_line = _mm_mul_ps(a_line, b_line); // r_line = a_line * b_line
    for (int j=1; j<4; j++) {
      a_line = _mm_load_ps(&a[j*4]); // a_line = vec4(column(a, j))
      b_line = _mm_set1_ps(b[i+j]);  // b_line = vec4(b[i][j])
                                     // r_line += a_line * b_line
      r_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), r_line);
    }
    _mm_store_ps(&r[i], r_line);     // r[i] = r_line
  }
}

class matrixSIMD4
{
    public:
        __m128 rows[4];
        // Matrix multiplication by devsh:
        //URL:
        // http://irrlicht.sourceforge.net/forum/viewtopic.php?f=9&t=50230&p=293604#p293604
        void mmul_devsh(const matrixSIMD4& other_a,const matrixSIMD4& other_b ){
                __m128 xmm4 = other_b.rows[0];
                __m128 xmm5 = other_b.rows[1];
                __m128 xmm6 = other_b.rows[2];
                __m128 xmm7 = other_b.rows[3];
                _MM_TRANSPOSE4_PS(xmm4,xmm5,xmm6,xmm7);

                __m128 xmm0 = other_a.rows[0];
                __m128 xmm1 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm4),_mm_mul_ps(xmm0,xmm5)); //(x_l,x_u,y_l,y_u)
                __m128 xmm2 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm6),_mm_mul_ps(xmm0,xmm7)); //(z_l,z_u,w_l,w_u)
                rows[0] = _mm_hadd_ps(xmm1,xmm2); //(x,y,z,w)

                xmm0 = other_a.rows[1];
                xmm1 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm4),_mm_mul_ps(xmm0,xmm5)); //(x_l,x_u,y_l,y_u)
                xmm2 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm6),_mm_mul_ps(xmm0,xmm7)); //(z_l,z_u,w_l,w_u)
                rows[1] = _mm_hadd_ps(xmm1,xmm2); //(x,y,z,w)

                xmm0 = other_a.rows[2];
                xmm1 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm4),_mm_mul_ps(xmm0,xmm5)); //(x_l,x_u,y_l,y_u)
                xmm2 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm6),_mm_mul_ps(xmm0,xmm7)); //(z_l,z_u,w_l,w_u)
                rows[2] = _mm_hadd_ps(xmm1,xmm2); //(x,y,z,w)

                xmm0 = other_a.rows[3];
                xmm1 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm4),_mm_mul_ps(xmm0,xmm5)); //(x_l,x_u,y_l,y_u)
                xmm2 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm6),_mm_mul_ps(xmm0,xmm7)); //(z_l,z_u,w_l,w_u)
                rows[3] = _mm_hadd_ps(xmm1,xmm2); //(x,y,z,w)
    }
};
/** Matrix multiplication like in irrlicht*/
void irr_mul(const float * m1, const float * m2, float * M){

    M[0] = m1[0]*m2[0] + m1[4]*m2[1] + m1[8]*m2[2] + m1[12]*m2[3];
    M[1] = m1[1]*m2[0] + m1[5]*m2[1] + m1[9]*m2[2] + m1[13]*m2[3];
    M[2] = m1[2]*m2[0] + m1[6]*m2[1] + m1[10]*m2[2] + m1[14]*m2[3];
    M[3] = m1[3]*m2[0] + m1[7]*m2[1] + m1[11]*m2[2] + m1[15]*m2[3];
    M[4] = m1[0]*m2[4] + m1[4]*m2[5] + m1[8]*m2[6] + m1[12]*m2[7];
    M[5] = m1[1]*m2[4] + m1[5]*m2[5] + m1[9]*m2[6] + m1[13]*m2[7];
    M[6] = m1[2]*m2[4] + m1[6]*m2[5] + m1[10]*m2[6] + m1[14]*m2[7];
    M[7] = m1[3]*m2[4] + m1[7]*m2[5] + m1[11]*m2[6] + m1[15]*m2[7];
    M[8] = m1[0]*m2[8] + m1[4]*m2[9] + m1[8]*m2[10] + m1[12]*m2[11];
    M[9] = m1[1]*m2[8] + m1[5]*m2[9] + m1[9]*m2[10] + m1[13]*m2[11];
    M[10] = m1[2]*m2[8] + m1[6]*m2[9] + m1[10]*m2[10] + m1[14]*m2[11];
    M[11] = m1[3]*m2[8] + m1[7]*m2[9] + m1[11]*m2[10] + m1[15]*m2[11];
    M[12] = m1[0]*m2[12] + m1[4]*m2[13] + m1[8]*m2[14] + m1[12]*m2[15];
    M[13] = m1[1]*m2[12] + m1[5]*m2[13] + m1[9]*m2[14] + m1[13]*m2[15];
    M[14] = m1[2]*m2[12] + m1[6]*m2[13] + m1[10]*m2[14] + m1[14]*m2[15];
    M[15] = m1[3]*m2[12] + m1[7]*m2[13] + m1[11]*m2[14] + m1[15]*m2[15];
}

void printMatrix(float * a){
    int i = 0;
    std::cout<<std::endl<<a[0]<<" "<<a[1]<<" "<<a[2]<<" "<<a[3]<<" "<<std::endl;
    std::cout<<a[4]<<" "<<a[5]<<" "<<a[6]<<" "<<a[7]<<" "<<std::endl;
    std::cout<<a[8]<<" "<<a[9]<<" "<<a[10]<<" "<<a[11]<<" "<<std::endl;
    std::cout<<a[12]<<" "<<a[13]<<" "<<a[14]<<" "<<a[15]<<" "<<std::endl;
}

void loadMatrix(float * a){

    std::ifstream myFile("identityMatrix.txt");  //opens .txt file

    if (!myFile.is_open()){  // check file is open, quit if not
        std::cout << "failed to open file\n";
        return ;
    }

    int i=0;
    float b;
    std::cout<<"entering\n"<<std::endl;
    while(myFile >> b){
        a[i++]=b;
        std::cout<<b<<" ";
    }

    myFile.close();
}

int main(){
    Matrix a,b,c;
    SMatrix x,y,z;
    matrixSIMD4 d1,d2,d3;
    int times = 10000000;

    loadMatrix(reinterpret_cast<float *>(&d1));
    loadMatrix(reinterpret_cast<float *>(&d2));
    loadMatrix(reinterpret_cast<float *>(&d3));

    loadMatrix(a.m);
    loadMatrix(b.m);
    loadMatrix(c.m);

    loadMatrix(reinterpret_cast<float *>(x.m));
    loadMatrix(reinterpret_cast<float *>(y.m));
    loadMatrix(reinterpret_cast<float *>(z.m));

    printMatrix(a.m);
    printMatrix(b.m);
    printMatrix(c.m);
    printMatrix(reinterpret_cast<float *>(&d1));
    printMatrix(reinterpret_cast<float *>(&d2));
    printMatrix(reinterpret_cast<float *>(&d3));
    printMatrix(reinterpret_cast<float *>(x.m));
    printMatrix(reinterpret_cast<float *>(y.m));
    printMatrix(reinterpret_cast<float *>(z.m));
    std::cout<<"------------"<<std::endl;

    auto t1 = Clock::now();
    auto t2 = Clock::now();

    assert( alignof(a)==alignof(b));
    assert( alignof(a)== 4);
    assert( alignof(x)==alignof(y));
    assert( alignof(x)== 16);

    t1 = Clock::now();
    for(int i=0; i<times; i++){
        d3.mmul_devsh(d1,d2);
        d1.mmul_devsh(d3,d2);
        d2.mmul_devsh(d1,d3);
    }
    t2 = Clock::now();

    STOP_TIMER("devsh SSE mat mul\t");

    t1 = Clock::now();
    for(int i=0; i<times; i++){
        mmul_sse(   reinterpret_cast<float *>(x.m),
                    reinterpret_cast<float *>(y.m),
                    reinterpret_cast<float *>(z.m));
        mmul_sse(   reinterpret_cast<float *>(z.m),
                    reinterpret_cast<float *>(y.m),
                    reinterpret_cast<float *>(x.m));
        mmul_sse(   reinterpret_cast<float *>(x.m),
                    reinterpret_cast<float *>(z.m),
                    reinterpret_cast<float *>(y.m));

    }
    t2 = Clock::now();

    STOP_TIMER("SSE matrix mul   \t");



    t1 = Clock::now();
    for(int i=0; i<times; i++){
        mmul(a.m,b.m,c.m);
        mmul(c.m,b.m,a.m);
        mmul(a.m,c.m,b.m);
    }
    t2 = Clock::now();


    STOP_TIMER("regularMatrix mul\t");

    t1 = Clock::now();
    for(int i=0; i<times; i++){
        irr_mul(a.m,b.m,c.m);
        irr_mul(c.m,b.m,a.m);
        irr_mul(a.m,c.m,b.m);
    }
    t2 = Clock::now();


    STOP_TIMER("irrMatrix mul\t");

    printMatrix(a.m);
    printMatrix(b.m);
    printMatrix(c.m);
    printMatrix(reinterpret_cast<float *>(x.m));
    printMatrix(reinterpret_cast<float *>(y.m));
    printMatrix(reinterpret_cast<float *>(z.m));
    printMatrix(reinterpret_cast<float *>(&d1));
    printMatrix(reinterpret_cast<float *>(&d2));
    printMatrix(reinterpret_cast<float *>(&d3));
    return 0;
}


