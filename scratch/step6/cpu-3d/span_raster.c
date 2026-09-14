/* CPU3D-FETCH-1: a span/edge-function z-buffer rasteriser over the ACTUAL Fetch robot triangles.
 * Reads the binary fetch_mesh_dump.py writes; per state: clear, rasterise every triangle with a
 * bbox-bounded edge-function scan, z-test, Gouraud-interpolate the normal, shade (headlight +
 * one directional light, Blinn-Phong), pack. Reports ns/triangle, ms/frame, fps, covered pixels.
 *   gcc -O3 -march=native -fno-math-errno -o span_raster span_raster.c -lm && ./span_raster dump.bin
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static inline uint8_t q8(float c){c=c<0?0:(c>1?1:c);return (uint8_t)(c*255.0f);}
int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage: span_raster dump.bin [reps] [mode: full|setup|bbox|nocull]\n");return 1;}
  const char*mode=argc>3?argv[3]:"full"; int cull=strcmp(mode,"nocull")!=0, do_scan=strcmp(mode,"setup")!=0, do_shade=strcmp(mode,"bbox")!=0;
  FILE*fp=fopen(argv[1],"rb"); int n,ntri,W; if(fread(&n,4,1,fp)!=1||fread(&ntri,4,1,fp)!=1||fread(&W,4,1,fp)!=1)return 2;
  int reps=argc>2?atoi(argv[2]):5;
  size_t per=(size_t)ntri*18; float*all=malloc(sizeof(float)*per*n); if(fread(all,sizeof(float),per*n,fp)!=per*n){fprintf(stderr,"short read\n");return 3;} fclose(fp);
  int N=W*W; float*zb=malloc(sizeof(float)*N); uint8_t*img=malloc(3*N);
  const float lx=0.0f,ly=0.0f,lz=1.0f;  /* directional light straight down, headlight toward camera ~ +z_cam */
  double tot=0; long covered=0,bboxpx=0,tri_seen=0; int frames=0;
  for(int r=0;r<reps;r++) for(int s=0;s<n;s++){
    const float*T=all+per*s; double t0=now();
    for(int i=0;i<N;i++)zb[i]=1e30f; memset(img,0,3*N);
    for(int t=0;t<ntri;t++){const float*a=T+18*t,*b=a+6,*c=a+12; tri_seen++;
      if(a[2]<=0||b[2]<=0||c[2]<=0)continue;
      float mnx=fminf(a[0],fminf(b[0],c[0])),mxx=fmaxf(a[0],fmaxf(b[0],c[0])),mny=fminf(a[1],fminf(b[1],c[1])),mxy=fmaxf(a[1],fmaxf(b[1],c[1]));
      int x0=(int)floorf(mnx),x1=(int)ceilf(mxx),y0=(int)floorf(mny),y1=(int)ceilf(mxy);
      if(x1<0||y1<0||x0>=W||y0>=W)continue; if(x0<0)x0=0;if(y0<0)y0=0;if(x1>=W)x1=W-1;if(y1>=W)y1=W-1;
      float area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]); if(fabsf(area)<1e-9f)continue;
      /* backface cull: MuJoCo/MJWarp draw front faces only; a counter-wound screen triangle is the back of the mesh */
      if(cull&&area<0)continue; float ia=1.0f/area; if(!do_scan)continue;
      for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++){bboxpx++; float px=x+0.5f,py=y+0.5f;
        float w0=((b[0]-px)*(c[1]-py)-(b[1]-py)*(c[0]-px))*ia,w1=((c[0]-px)*(a[1]-py)-(c[1]-py)*(a[0]-px))*ia,w2=1.0f-w0-w1;
        if(w0<0||w1<0||w2<0)continue; float z=w0*a[2]+w1*b[2]+w2*c[2]; int i=y*W+x; if(z>=zb[i])continue; zb[i]=z; covered++; if(!do_shade)continue;
        float nx=w0*a[3]+w1*b[3]+w2*c[3],ny=w0*a[4]+w1*b[4]+w2*c[4],nz=w0*a[5]+w1*b[5]+w2*c[5]; float nl=sqrtf(nx*nx+ny*ny+nz*nz); if(nl>0){nx/=nl;ny/=nl;nz/=nl;}
        float ndl=nx*lx+ny*ly+nz*lz; if(ndl<0)ndl=0; float hl=nz>0?nz:0; float spec=0.5f*powf(ndl,64.0f)*0.3f+0.5f*powf(hl,64.0f)*0.5f;
        float base=0.8f*(0.2f+0.1f+0.8f*ndl+0.4f*hl)+spec; uint8_t*q=img+3*i; q[0]=q8(base);q[1]=q8(base*0.9f);q[2]=q8(base*0.8f);}
    }
    tot+=now()-t0; frames++;
  }
  double dt=tot/frames; long chk=0; for(int i=0;i<3*N;i++)chk+=img[i];
  printf("{\"mode\":\"%s\",\"cull\":%d,\"ntri\":%d,\"states\":%d,\"reps\":%d,\"res\":%d,\"ms_per_frame\":%.3f,\"fps_1core\":%.1f,\"ns_per_tri\":%.1f,\"covered_px_per_frame\":%.0f,\"bbox_px_per_frame\":%.0f,\"chk\":%ld}\n",
    mode,cull,ntri,n,reps,W,dt*1e3,1.0/dt,dt/ntri*1e9,(double)covered/frames,(double)bboxpx/frames,chk%1000);
  return 0;
}
