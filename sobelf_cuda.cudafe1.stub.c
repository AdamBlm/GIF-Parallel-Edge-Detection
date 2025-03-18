#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "sobelf_cuda.fatbin.c"
extern void __device_stub__Z16grayscale_kernelP5pixelii(pixel *, int, int);
extern void __device_stub__Z11blur_kernelP5pixelS0_iii(pixel *, pixel *, int, int, int);
extern void __device_stub__Z24check_convergence_kernelP5pixelS0_iiiPi(pixel *, pixel *, int, int, int, int *);
extern void __device_stub__Z12sobel_kernelP5pixelS0_ii(pixel *, pixel *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z16grayscale_kernelP5pixelii(pixel *__par0, int __par1, int __par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 12UL);__cudaLaunch(((char *)((void ( *)(pixel *, int, int))grayscale_kernel)));}
# 163 "src/sobelf_cuda.cu"
void grayscale_kernel( pixel *__cuda_0,int __cuda_1,int __cuda_2)
# 164 "src/sobelf_cuda.cu"
{__device_stub__Z16grayscale_kernelP5pixelii( __cuda_0,__cuda_1,__cuda_2);
# 178 "src/sobelf_cuda.cu"
}
# 1 "sobelf_cuda.cudafe1.stub.c"
void __device_stub__Z11blur_kernelP5pixelS0_iii( pixel *__par0,  pixel *__par1,  int __par2,  int __par3,  int __par4) {  __cudaLaunchPrologue(5); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaLaunch(((char *)((void ( *)(pixel *, pixel *, int, int, int))blur_kernel))); }
# 181 "src/sobelf_cuda.cu"
void blur_kernel( pixel *__cuda_0,pixel *__cuda_1,int __cuda_2,int __cuda_3,int __cuda_4)
# 182 "src/sobelf_cuda.cu"
{__device_stub__Z11blur_kernelP5pixelS0_iii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 239 "src/sobelf_cuda.cu"
}
# 1 "sobelf_cuda.cudafe1.stub.c"
void __device_stub__Z24check_convergence_kernelP5pixelS0_iiiPi( pixel *__par0,  pixel *__par1,  int __par2,  int __par3,  int __par4,  int *__par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 32UL); __cudaLaunch(((char *)((void ( *)(pixel *, pixel *, int, int, int, int *))check_convergence_kernel))); }
# 242 "src/sobelf_cuda.cu"
void check_convergence_kernel( pixel *__cuda_0,pixel *__cuda_1,int __cuda_2,int __cuda_3,int __cuda_4,int *__cuda_5)
# 244 "src/sobelf_cuda.cu"
{__device_stub__Z24check_convergence_kernelP5pixelS0_iiiPi( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 259 "src/sobelf_cuda.cu"
}
# 1 "sobelf_cuda.cudafe1.stub.c"
void __device_stub__Z12sobel_kernelP5pixelS0_ii( pixel *__par0,  pixel *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(pixel *, pixel *, int, int))sobel_kernel))); }
# 262 "src/sobelf_cuda.cu"
void sobel_kernel( pixel *__cuda_0,pixel *__cuda_1,int __cuda_2,int __cuda_3)
# 263 "src/sobelf_cuda.cu"
{__device_stub__Z12sobel_kernelP5pixelS0_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 329 "src/sobelf_cuda.cu"
}
# 1 "sobelf_cuda.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T13) {  __nv_dummy_param_ref(__T13); __nv_save_fatbinhandle_for_managed_rt(__T13); __cudaRegisterEntry(__T13, ((void ( *)(pixel *, pixel *, int, int))sobel_kernel), _Z12sobel_kernelP5pixelS0_ii, (-1)); __cudaRegisterEntry(__T13, ((void ( *)(pixel *, pixel *, int, int, int, int *))check_convergence_kernel), _Z24check_convergence_kernelP5pixelS0_iiiPi, (-1)); __cudaRegisterEntry(__T13, ((void ( *)(pixel *, pixel *, int, int, int))blur_kernel), _Z11blur_kernelP5pixelS0_iii, (-1)); __cudaRegisterEntry(__T13, ((void ( *)(pixel *, int, int))grayscale_kernel), _Z16grayscale_kernelP5pixelii, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
