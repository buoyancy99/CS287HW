//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright 2009-2015 Roboti LLC.  //
//-----------------------------------//
#pragma once

// includes
#include "mex.h"
#include "mujoco.h"
#include "mjxmacro.h"

// type definitions
#ifdef mjUSEDOUBLE
	#define mxMUJOCO_CLASS mxDOUBLE_CLASS
#else
	#define mxMUJOCO_CLASS mxSINGLE_CLASS // note: has not been tested
#endif

// global constant: number of threads
const int NTHREAD = 8;

// global variables
extern mjModel* model;
extern mjData* data;
extern mjData* DATA[NTHREAD];

// error and memory handlers
void mju_MATLAB_error(const char* msg);
void mju_MATLAB_warning(const char* msg);
void* mju_MATLAB_malloc(size_t sz);
void mju_MATLAB_free(void* ptr);


// MATLAB => Mujoco copy
template <typename T>
static void mat2mjc(T* mjc, int nr, int nc, const double* mat)
{
	for( int r=0; r<nr; r++ )
		for( int c=0; c<nc; c++ )
			mjc[c+r*nc] = (T)mat[r+c*nr];
}

// Mujoco => MATLAB copy
template <typename T>
static void mjc2mat(double* mat, int nr, int nc, const T* mjc)
{
	for( int r=0; r<nr; r++ )
		for( int c=0; c<nc; c++ )
			mat[r+c*nr] = (double)mjc[c+r*nc];
}

// visualiztion functions
int visualize_init();
void visualize_close();

// help text
static const char* _mj_help =
"\n"
" mj: mex interface to MuJoCo\n"
"    mj uses persistent memory to maintain local instances of mjModel and mjData\n\n"

" Usage:\n"
"    mj('command', ...)\n\n"

" Basic information:\n"
"   (no command)           display help\n"
"   version                get version of the MuJoCo engine\n"
"   schema                 display the MJCF schema\n"
"   activate licenseFile   activate license\n"
"   deactivate             deactivate license\n\n"

"------(the rest can only be called after MuJoCo license has been activated)------\n\n"

" Initialization:\n"        
"   clear                  clear model and data, unlock mex\n"
"   ismodel                return 1 if a model is loaded, 0 otherwise\n"
"   load filename          load model, lock mex\n\n"

"------(the rest can only be called after a model is loaded)------\n\n"

" Vizualization:\n"
"   plot                   open visualization window\n\n"

" Main computation:\n"
"   reset                  reset mjData\n"
"   kinematics             compute kinematics only\n"
"   forward                compute forward dynamics\n"
"   inverse                compute inverse dynamics\n"
"   step                   advance simulation\n"
"   y = step(x,u)          vectorized step, x and u are matrices \n"
"   step1                  advance simulation in 2 phases: before ctrl is set\n"
"   step2                  advance simulation in 2 phases: after ctrl is set\n\n"

" I/O commands:\n"
"   size                   get model dimensions\n"        
"   getmodel               get mjModel structure (without meshes)\n"
"   setmodel mod           set modifiable fields of mjModel from mod struct\n"        
"   setmodelfield field,value,...  set specified fields of mjModel\n"        
"   getmesh                get mesh data from mjModel\n"        
"   getoption              get mjOption from mjModel\n"
"   setoption opt          set mjOption from opt struct\n"        
"   getdata                get all fields of mjData\n"
"   get field,...          get specified fields from mjData\n"
"   set field,value,...    set specified fields of mjData, if modifiable\n\n"        

"     call 'setmodelfield', 'get' and 'set' with no inputs for detailed help\n\n"

" Object Information:\n"
"   getname type id        get the name of the object given its type and id (0-base)\n"
"   getid type name        get the id of the object given its type and name\n\n"
"     valid object types:  body, joint, geom, site, mesh, tendon, actuator,\n"
"                          constraint, custom\n\n"

" Saving:\n"
"   save filename          save binary model to file\n"        
"   printmodel filename    save mjModel to text file\n"
"   printdata filename     save mjData to text file\n\n"         
        
" Support functions:\n"
"   add   x  dx            y = x + dx, take quaterions into account\n"
"   diff  y  x             dx = y - x, take quaterions into account\n\n"

" Jacobians:\n"
"   jacpoint pnt b_ind     jacobian of point (in global coordinates) with respect to\n"
"                            body with specified index\n"
"   jacaxis axis b_ind     jacobian of axis (unit vector, in global coordinates)\n"
"   jacbody [b_ind,...]    jacobian of body frame center\n"
"   jacbodycom [b_ind,...] jacobian of body center of mass\n"
"   jacsite [s_ind,...]    jacobian of site\n"
"   jacgeom [g_ind,...]    jacobian of geom\n\n"
"     jacXXX returns two 3-by-nv matrices: the translation and rotation jacobians\n"
"     when multiple objects are specified the outputs are stacked: 3*nobj-by-nv\n\n"
;