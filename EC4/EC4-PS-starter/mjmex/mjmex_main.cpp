//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright 2009-2015 Roboti LLC.  //
//-----------------------------------//
#include "mjmex.h"
#include "stdio.h"
#include "string.h"
#include "math.h"

#ifndef WIN32
    #define _strcmpi strcasecmp
#endif

// global model and data
mjModel* model = 0;
mjData* data = 0;

// global data array for multithreading
mjData* DATA[NTHREAD];

// initialization
static bool initialized = false;

//------------------------- error handlers ----------------------------------------------

// write message to logfile, call MATLAB with mexErrMsgTxt
void mju_MATLAB_error(const char* msg)
{
    if( data )
        data->pstack = 0;       // reset stack
    mexErrMsgTxt(msg);
}


// write message to logfile, call MATLAB with mexWarnMsgTxt
void mju_MATLAB_warning(const char* msg)
{
    mexWarnMsgTxt(msg);
}


// allocate memory with mxMalloc, make persistent
void* mju_MATLAB_malloc(size_t sz)
{
    void* ptr = mxMalloc(sz);
    mexMakeMemoryPersistent(ptr);
    return ptr;
}


// free memory with mxFree
void mju_MATLAB_free(void* ptr)
{
    mxFree(ptr);
}


// check size of numeric input
static void checkNumeric(const mxArray* mx, const char* name, int sz0, int sz1)
{
    char msg[100];

    if( !mx )
    {
        sprintf(msg, "%s: missing numeric argument", name);
        mexErrMsgTxt(msg);
    }

    if( mxGetNumberOfDimensions(mx)!=2 )
    {
        sprintf(msg, "%s: numeric argument has %d dimensions, should be 2",
            name, (int)mxGetNumberOfDimensions(mx));
        mexErrMsgTxt(msg);
    }

    if( mxGetClassID(mx)!= mxDOUBLE_CLASS )
    {
        sprintf(msg, "%s: expected class DOUBLE", name);
        mexErrMsgTxt(msg);
    }

    const mwSize* sz = mxGetDimensions(mx);
    if( (sz[0]!=sz0 && sz0>=0) || (sz[1]!=sz1 && sz1>=0) )
    {
        sprintf(msg, "%s: expected %d-by-%d, got %d-by-%d",
            name, sz0, sz1, (int)sz[0], (int)sz[1]);
        mexErrMsgTxt(msg);
    }
}


//------------------------ copy data between MATLAB and Mujoco --------------------------



// MATLAB => Mujoco: structure field, numeric
template <typename T>
bool mx2mjc(T* mj, const mxArray* arg, const int nr, const int nc, const char* name)
{
    // get field and check
    const mxArray* mx = mxGetField(arg, 0, name);
    if( !mx )
        return false;
    checkNumeric(mx, name, nr, nc);

    // copy data
    mat2mjc(mj, nr, nc, mxGetPr(mx));

    return true;
}



// MATLAB => Mujoco: structure field, string
bool mx2mjcStr(char* str, const mxArray* arg, const char* name)
{
    // get field, check type
    const mxArray* mx = mxGetField(arg, 0, name);
    if( !mx )
        return false;
    if( mxGetClassID(mx)!= mxCHAR_CLASS )
        mexErrMsgTxt("string expected");

    // copy string
    mxGetString(mx, str, 100);

    return true;
}



// Mujoco => MATLAB: structure field, numeric
template <typename T>
void mjc2mx(mxArray* out, const T* mj, const int nr, const int nc,
            const char* name, int id=0)
{
    // return if no data; empty matrix assigned by default
    if( !nr || !nc )
        return;

    // check field name
    if( mxGetFieldNumber(out, name)==-1 )
    {
        mexPrintf("field name  %s  unrecognized\n", name);
        mexErrMsgTxt("error copying data from MuJoCo to Matlab");
    }

    // create field
    mxSetField(out, id, name, mxCreateDoubleMatrix(nr, nc, mxREAL));

    // copy data
    double* ptr = mxGetPr(mxGetField(out, id, name));
    mjc2mat(ptr, nr, nc, mj);
}


// dummy overload with mjContact, to prevent casting error
void mjc2mx(mxArray* out, const mjContact* mj, const int nr, const int nc,
        const char* name, int id=0) {}


// Mujoco => MATLAB: string tables
void setStrings(mxArray* out, int size, const int* adr, const char* data, 
        const char* name)
{
    if ( !size )
        return;
    // check field name
    if( mxGetFieldNumber(out, name)==-1 )
    {
        mexPrintf("field name  %s  unrecognized\n", name);
        mexErrMsgTxt("error copying data from MuJoCo to Matlab");
    }

    // prepare pointers
    if( size>1000 )
        mexErrMsgTxt("over 1000 names, increase preallocated limit");
    const char *pstr[1000];
    for( int i=0; i<size; i++ )
        pstr[i] = data + adr[i];

    // create string table, assign
    mxSetField(out, 0, name, mxCreateCharMatrixFromStrings(size, pstr));
}



//----------------------- get mjOption, mjModel structures; fields of mjData ------------

// get mjOption (from mjModel)
mxArray* getOption(void)
{
    // create Matlab structure
    int size = 0;
    #define X(type, name) size++;
        MJOPTION_SCALARS
        MJOPTION_VECTORS
    #undef X

    const char* name[30] = {
    #define X(type, name) #name,
        MJOPTION_SCALARS
    #undef X
    #define X(name, num) #name,
        MJOPTION_VECTORS
    #undef X
    };

    // create option structure
    mxArray* out = mxCreateStructMatrix(1, 1, size, name);

    // fill structure with data
    #define X(type, name) mjc2mx(out, &model->opt.name,     1, 1, #name);
        MJOPTION_SCALARS
    #undef X
    #define X(name, num) mjc2mx(out, model->opt.name,       1, num, #name);
        MJOPTION_VECTORS
    #undef X
    return out;
}


// get mesh data (from mjModel)
mxArray* getMesh(void)
{
    // create Matlab structure
    const int size = 8;
    const char* name[size] = {
        "faceadr",
        "facenum",
        "vertadr",
        "vertnum",
        "graphadr",
        "vert",
        "face",
        "graph"
    };

    // create option structure
    mxArray* out = mxCreateStructMatrix(1, 1, size, name);

    // fill structure with data
    mjc2mx(out, model->mesh_faceadr,    model->nmesh,       1, "faceadr");
    mjc2mx(out, model->mesh_facenum,    model->nmesh,       1, "facenum");
    mjc2mx(out, model->mesh_vertadr,    model->nmesh,       1, "vertadr");
    mjc2mx(out, model->mesh_vertnum,    model->nmesh,       1, "vertnum");
    mjc2mx(out, model->mesh_graphadr,   model->nmesh,       1, "graphadr");
    mjc2mx(out, model->mesh_vert,       model->nmeshvert,   3, "vert");
    mjc2mx(out, model->mesh_face,       model->nmeshface,   3, "face");
    mjc2mx(out, model->mesh_graph,      model->nmeshgraph,  1, "graph");
    return out;
}


// count ints in mjModel
static int getnint(void)
{
    int cnt = 0;

    #define X(name) cnt++;
        MJMODEL_INTS
    #undef X

    return cnt;
}

// count pointers in mjModel
static int getnptr(void)
{
    int cnt = 0;

    #define X(type, name, nr, nc) cnt++;
        MJMODEL_POINTERS
    #undef X

    return cnt;
}

// get mjModel
mxArray* getModel(void)
{
    mjModel* m = model;

    // prepare variable sizes needed by xmacro
    int nuser_body = model->nuser_body;
    int nuser_jnt = model->nuser_jnt;
    int nuser_geom = model->nuser_geom;
    int nuser_site = model->nuser_site;
    int nuser_tendon = model->nuser_tendon;
    int nuser_actuator = model->nuser_actuator;
    int nuser_sensor = model->nuser_sensor;
    int nq = model->nq;
    int nv = model->nv;
    int na = model->na;

    // create array of names
    const char* names[1000] = {
        "model_name",
        #define X(name) #name,
            MJMODEL_INTS
        #undef X
        #define X(type, name, nr, nc) #name,
            MJMODEL_POINTERS
        #undef X
        "body_names",
        "joint_names",
        "geom_names",
        "site_names",
        "camera_names",
        "light_names",
        "mesh_names",
        "hfield_names",
        "texture_names",
        "material_names",
        "equality_names",
        "tendon_names",
        "actuator_names",
        "sensor_names",
        "numeric_names",
        "text_names"
    };

    // compute size
    int size = 1 + getnint() + getnptr() + 16;
    if( size>1000 )
        mexErrMsgTxt("over 1000 fields, increase preallocated limit");

    // create structure
    mxArray* out = mxCreateStructMatrix(1, 1, size, names);

    // === fill structure with data from MuJoCo ===

    // integers:
    #define X(name) mjc2mx(out, &model->name, 1, 1, #name);
        MJMODEL_INTS
    #undef X

    // pointers (not including meshes and strings):
    #define X(type, name, nr, nc) \
        if( strncmp(#name, "mesh_", 5) && strncmp(#name, "name", 4) )\
            mjc2mx(out, model->name, model->nr, nc, #name);\

        MJMODEL_POINTERS
    #undef X

    // strings:
    mxSetField(out, 0, "model_name", mxCreateString(model->names));
    setStrings(out, model->nbody,    model->name_bodyadr,     model->names, "body_names");
    setStrings(out, model->njnt,     model->name_jntadr,      model->names, "joint_names");
    setStrings(out, model->ngeom,    model->name_geomadr,     model->names, "geom_names");
    setStrings(out, model->nsite,    model->name_siteadr,     model->names, "site_names");   
    setStrings(out, model->ncam,     model->name_camadr,      model->names, "camera_names");
    setStrings(out, model->nlight,   model->name_lightadr,    model->names, "light_names");
    setStrings(out, model->nmesh,    model->name_meshadr,     model->names, "mesh_names");
    setStrings(out, model->nhfield,  model->name_hfieldadr,   model->names, "hfield_names");
    setStrings(out, model->ntex,     model->name_texadr,      model->names, "texture_names");
    setStrings(out, model->nmat,     model->name_matadr,      model->names, "material_names");
    setStrings(out, model->neq,      model->name_eqadr,       model->names, "equality_names");
    setStrings(out, model->ntendon,  model->name_tendonadr,   model->names, "tendon_names");
    setStrings(out, model->nu,       model->name_actuatoradr, model->names, "actuator_names");
    setStrings(out, model->nsensor,  model->name_sensoradr,   model->names, "sensor_names");
    setStrings(out, model->nnumeric, model->name_numericadr,  model->names, "numeric_names");
    setStrings(out, model->ntext,    model->name_textadr,     model->names, "text_names");

    // remove empty fields
    for ( int i=size-1; i>=0; i--)
        if ( !mxGetFieldByNumber(out, 0, i) || !mxGetNumberOfElements(mxGetFieldByNumber(out, 0, i)) )
            mxRemoveField(out, i);

    return out;
}



// get contacts (from mjData)
mxArray* getContact(void)
{
    // create Matlab structure
    const int size = 13;
    const char* name[size] = {
        "dim",
        "geom1",
        "geom2",
        "dist",
        "pos",
        "frame",
        "friction",
        "solref",
        "solimp",
        "includemargin",        
        "exclude",
        "force",
        "jac"
    };

    // create contact structure
    mxArray* out = mxCreateStructMatrix(1, data->ncon, size, name);

    // fill structure with data
    for( int i=0; i<data->ncon; i++ )
        if( data->contact[i].efc_address >= 0 )
        {
            // copy explicit data
            mjc2mx(out, &data->contact[i].dim,            1,       1, "dim", i);
            mjc2mx(out, &data->contact[i].geom1,          1,       1, "geom1", i);
            mjc2mx(out, &data->contact[i].geom2,          1,       1, "geom2", i);
            mjc2mx(out, data->contact[i].solref,          mjNREF,  1, "solref", i);
            mjc2mx(out, data->contact[i].solimp,          mjNIMP,  1, "solimp", i);
            mjc2mx(out, &data->contact[i].dist,           1,       1, "dist", i);
            mjc2mx(out, &data->contact[i].includemargin,  1,       1, "includemargin", i);
            mjc2mx(out, &data->contact[i].exclude,        1,       1, "exclude", i);
            mjc2mx(out, data->contact[i].pos,             3,       1, "pos", i);
            mjc2mx(out, data->contact[i].frame,           3,       3, "frame", i);
            mjc2mx(out, data->contact[i].friction,        5,       1, "friction", i);

            // get contact/limit force from efc_f
            mjtNum force[6] = {0,0,0,0,0,0};
            mju_copy(force, data->efc_force + data->contact[i].efc_address, data->contact[i].dim);
            mjc2mx(out, force, 6, 1, "force", i);

            // get Jacobian efc_J
            int nv = model->nv;
            int _mark = data->pstack;
            mjtNum* J = mj_stackAlloc(data, data->contact[i].dim*nv);
            mju_copy(J, data->efc_J + nv*data->contact[i].efc_address, nv*data->contact[i].dim);
            mjc2mx(out, J, data->contact[i].dim, nv, "jac", i);
            data->pstack = _mark;
        }

    return out;
}



// get one field of mjData
template <typename T>
mxArray* getOneField(int nr, int nc, T* data)
{
    // create MATLAB matrix, copy
    mxArray* out = mxCreateDoubleMatrix(nr, nc, mxREAL);
    mjc2mat(mxGetPr(out), nr, nc, data);

    return out;
}

void helpGet(void)
{
    mexPrintf(
    " Get multiple fields of mjData. usage:\n"
    " [val1,val2,...] = mj('get','field1','field2',...)\n"
    " Valid field names:\n");
    #define X(type, name, nr, nc)\
        mexPrintf("    " #name "\n");
        MJDATA_POINTERS
    #undef X
    #define X(type, name)\
        mexPrintf("    " #name "\n");
        MJDATA_SCALAR
    #undef X
    #define X(type, name, nr, nc)\
        mexPrintf("    " #name "\n");
        MJDATA_VECTOR
    #undef X
}

// dummy overload, to prevent casting error
mxArray* getOneField(int nr, int nc, mjContact* data)
{
    mxArray* out = mxCreateDoubleMatrix(1, 1, mxREAL);
    return out;
}

// get specified fields of mjData
void getField(int nout, mxArray* pout[], int nin, const mxArray* pin[])
{
    // no inputs: print help
    if( nin == 1 )
    {
        helpGet();
        return;
    }
    
    if( !model )
        mexErrMsgTxt("model has not been loaded");    
    
    char field[100];

    // precompute sizes
    int nu = model->nu;
    int nefc = data->nefc;
    int ncon = data->ncon;
    int nA = data->nefc*data->nefc;

    // prepare variable sizes needed by xmacro
    mjModel* m = model;
    int nv = m->nv, nemax = m->nemax, njmax = m->njmax;

    // loop over input/outputs
    for( int i=0; i<mju_max(1, mju_min(nout, nin-1)); i++ )
    {
        // get field name
        if( mxGetClassID(pin[i+1])!=mxCHAR_CLASS )
            mexErrMsgTxt("args must be field names");
        mxGetString(pin[i+1], field, 100);


        // === return specified field
        if ( 0 )
            mexErrMsgTxt("dummy if() clause");

        // scalars
        #define X(type, name) \
        else if( !strcmp(field, #name) )\
            pout[i] = getOneField(1, 1, &data->name);
        MJDATA_SCALAR
        #undef X

        // vectors
        #define X(type, name, nr, nc) \
        else if( !strcmp(field, #name) )\
            pout[i] = getOneField(nr, nc, data->name);
        MJDATA_VECTOR
        #undef X

        // unpack inertia matrices
        else if( !strcmp(field, "qM") || !strcmp(field, "qLD") )
        {
            int _mark = data->pstack;
            mjtNum* mat = mj_stackAlloc(data, model->nv*model->nv);
            mj_fullM(model, mat, !strcmp(field, "qM") ? data->qM : data->qLD);
            pout[i] = getOneField(model->nv, model->nv, mat);
            data->pstack = _mark;
        }

        // contact structure
        else if( !strcmp(field, "contact") )
            pout[i] = getContact();

        // buffer variables
        #define X(type, name, nr, nc) \
        else if( !strcmp(field, #name) )\
            pout[i] = getOneField(model->nr, nc, data->name);
        MJDATA_POINTERS
        #undef X

        // otherwise
        else
        {
            strcat(field, ": mjData does not have such field");
            mexErrMsgTxt(field);
        }
    }
}


// count element in mjData
static int getndata(void)
{
    int cnt = 0;

    #define X(type, name, nr, nc) cnt++;
        MJDATA_POINTERS
    #undef X

    #define X(type, name) cnt++;
        MJDATA_SCALAR
    #undef X

    #define X(type, name, nr, nc) cnt++;
        MJDATA_VECTOR
    #undef X

    return cnt;
}

// get all fields mjData
mxArray* getData()
{
    // precompute sizes
    int nu = model->nu;
    int nefc = data->nefc;
    int ncon = data->ncon;
    int nA = data->nefc*data->nefc;

    // prepare variable sizes needed by xmacro
    mjModel* m = model;
    int nv = m->nv, nemax = m->nemax, njmax = m->njmax;

    // create array of names
    const char* names[1000] = {
        #define X(type, name) #name,
            MJDATA_SCALAR
        #undef X
        #define X(type, name, nr, nc) #name,
            MJDATA_POINTERS
        #undef X
        #define X(type, name, nr, nc) #name,
            MJDATA_VECTOR
        #undef X
    };

    // compute size
    int size = getndata();
    if( size>1000 )
        mexErrMsgTxt("over 1000 fields, increase preallocated limit");

    // create structure
    mxArray* out = mxCreateStructMatrix(1, 1, size, names);

    // scalars:
    #define X(type, name) \
        mjc2mx(out, &data->name, 1, 1, #name);
        MJDATA_SCALAR
    #undef X

    // pointers:
    #define X(type, name, nr, nc) \
        mjc2mx(out, data->name, model->nr, nc, #name);
        MJDATA_POINTERS
    #undef X

    // vector:
    #define X(type, name, nr, nc) \
        if( strcmp(#name, "qM") && strcmp(#name, "qLD") ) \
            mjc2mx(out, data->name, nr, nc, #name);
        MJDATA_VECTOR
    #undef X

    // mass matrix and its square root
    int _mark = data->pstack;
    mjtNum* mat = mj_stackAlloc(data, model->nv*model->nv);
    mj_fullM(model, mat, data->qM);
    mjc2mx(out, mat, model->nv, model->nv, "qM");
    mj_fullM(model, mat, data->qLD);
    mjc2mx(out, mat, model->nv, model->nv, "qLD");
    data->pstack = _mark;

    // contact structure
    mxSetField(out, 0, "contact", getContact());

    // find and remove empty fields
    for ( int i=size-1; i>=0; i--)
        if ( !mxGetFieldByNumber(out, 0, i) || !mxGetNumberOfElements(mxGetFieldByNumber(out, 0, i)) )
            mxRemoveField(out, i);

    return out;
}


//----------------- set mjOption, mjModel; fields of mjData -----------------------------

// set option (inside mjModel)
void setOption(const mxArray* arg)
{
    #define X(type, name) mx2mjc(&model->opt.name, arg, 1, 1, #name);
        MJOPTION_SCALARS
    #undef X
    #define X(name, num) mx2mjc(model->opt.name, arg, 1, num, #name);
        MJOPTION_VECTORS
    #undef X
}


// set model
void setModel(const mxArray* arg)
{
    // prepare variable sizes needed by xmacro
    int nuser_body      = model->nuser_body;
    int nuser_jnt       = model->nuser_jnt;
    int nuser_geom      = model->nuser_geom;
    int nuser_site      = model->nuser_site;
    int nuser_tendon    = model->nuser_tendon;
    int nuser_actuator  = model->nuser_actuator;
    int nuser_sensor    = model->nuser_sensor;
    int nq              = model->nq;
    int nv              = model->nv;
    int na              = model->na;

    #define X(type, name, nr, nc) \
    if( !strcmp(#type, "mjtNum") ) \
        mx2mjc(model->name, arg, model->nr, nc, #name);\

    MJMODEL_POINTERS
    #undef X

}

void helpSetField(int nout, mxArray* pout[])
{
    if( nout == 0) // no outputs, just display help text
    {
        mexPrintf(
        " Set multiple fields of mjModel to specified values. Usage:\n"
        " mj('setmodelfield', 'field1', value1, 'field2', value2,...)\n"
        " Valid field names:\n");
        #define X(type, name, nr, nc)\
            if( !strcmp(#type, "mjtNum") ) mexPrintf("    " #name "\n");
            MJMODEL_POINTERS
        #undef X
    }
    else // output cell array of settable fields
    {
        // count settable fields
        int count=0;
        #define X(type, name, nr, nc)\
            if( !strcmp(#type, "mjtNum") ) count++;
            MJMODEL_POINTERS
        #undef X    
                
        // make cell array
        pout[0] = mxCreateCellMatrix(count, 1);
        
        // set field names
        count=0;
        #define X(type, name, nr, nc)\
            if( !strcmp(#type, "mjtNum") ) mxSetCell(pout[0], count++, mxCreateString(#name));
            MJMODEL_POINTERS
        #undef X                
    }
}


// set one numeric field (of mjModel or mjData)
template <typename T>
void setOneField(const double* ptr, const mwSize* sz, T* field, int nr, int nc)
{
    // check dimensions
    if( sz[0]!=nr || sz[1]!=nc )
    {
        char err[100];
        sprintf(err, "size mismatch: expected %d-by-%d, got %d-by-%d",
            nr, nc, (int)sz[0], (int)sz[1]);
        mexErrMsgTxt(err);
    }

    // empty: return
    if( !nr || !nc )
        return;

    // copy and transpose
    mat2mjc(field, nr, nc, ptr);
}


// dummy overload with mjContact to prevent casting error
void setOneField(const double* ptr, const mwSize* sz, mjContact* field, int nr, int nc) {}

// set specified fields of model
void setModelField(int nout, mxArray* pout[], int nin, const mxArray* pin[])
{
    // no inputs: print help
    if( nin == 1 )
    {
        helpSetField(nout, pout);
        return;
    }    
    
    if( !model )
        mexErrMsgTxt("model has not been loaded");    

    // prepare variable sizes needed by xmacro
    int nuser_body      = model->nuser_body;
    int nuser_jnt       = model->nuser_jnt;
    int nuser_geom      = model->nuser_geom;
    int nuser_site      = model->nuser_site;
    int nuser_tendon    = model->nuser_tendon;
    int nuser_actuator  = model->nuser_actuator;
    int nuser_sensor    = model->nuser_sensor;
    int nq              = model->nq;
    int nv              = model->nv;
    int na              = model->na;
    int nu              = model->nu;    

    // check input arguments
    if( (nin-1)%2 )
        mexErrMsgTxt("list of (name,value) pairs expected");

    // loop over (name,value) pairs
    for( int i=0; i<(nin-1)/2; i++ )
    {
        // check input types
        if( mxGetClassID(pin[2*i+1]) != mxCHAR_CLASS ||
            mxGetClassID(pin[2*i+2]) != mxDOUBLE_CLASS )
            mexErrMsgTxt("(string, double) expected");

        // get field name
        char field[100];
        mxGetString(pin[2*i+1], field, 100);

        // get/check dimensions, get data ptr
        if( mxGetNumberOfDimensions(pin[2*i+2])!=2 )
            mexErrMsgTxt("2D array expected");
        const mwSize* sz = mxGetDimensions(pin[2*i+2]);
        const double* ptr =  mxGetPr(pin[2*i+2]);

        // find field, set
        #define X(type, name, nr, nc) \
            if( !strcmp(field, #name) && !strcmp("mjtNum", #type)) {\
                setOneField(ptr, sz, model->name, model->nr, nc);\
                continue;\
            }

            MJMODEL_POINTERS
        #undef X

        strcat(field, ": no such field or field cannot be set");
        mexErrMsgTxt(field);
    }
}


void helpSet(void)
{
    mexPrintf(
    " Set multiple fields of mjData to specified values. Usage:\n"
    " mj('set', 'field1', value1, 'field2', value2, ...)\n"
    " Valid field names:\n");
    #define X(type, name)\
        if ( !strcmp("mjtNum", #type) )\
            mexPrintf("   " #name "\n");
        MJDATA_SCALAR
    #undef X
    #define X(type, name, nr, nc)\
        if ( !strcmp("mjtNum", #type) )\
            mexPrintf("   " #name "\n");
        MJDATA_POINTERS
        MJDATA_VECTOR
    #undef X
}


// set specified fields of mjData
void setDataField(int nin, const mxArray* pin[])
{
    // no inputs, print help
    if( nin==1 )
    {
        helpSet();
        return;
    }    
    
    if( !model )
        mexErrMsgTxt("model has not been loaded");    
    
    // precompute sizes
    int ne = data->ne;
    int nu = model->nu;
    int nefc = data->nefc;
    int ncon = data->ncon;
    int nA = data->nefc*data->nefc;
    int nwrp = 0;
    for( int i=0; i<model->ntendon; i++ )
        nwrp += data->ten_wrapnum[i];

    // check input arguments
    if( (nin-1)%2 )
        mexErrMsgTxt("list of (name,value) pairs expected");

    // loop over (name,value) pairs
    for( int i=0; i<(nin-1)/2; i++ )
    {
        // check input types
        if( mxGetClassID(pin[2*i+1]) != mxCHAR_CLASS ||
            mxGetClassID(pin[2*i+2]) != mxDOUBLE_CLASS )
            mexErrMsgTxt("(string, double) expected");

        // get field name
        char field[100];
        mxGetString(pin[2*i+1], field, 100);

        // get/check dimensions, get data ptr
        if( mxGetNumberOfDimensions(pin[2*i+2])!=2 )
            mexErrMsgTxt("2D array expected");
        const mwSize* sz = mxGetDimensions(pin[2*i+2]);
        const double* ptr =  mxGetPr(pin[2*i+2]);

        // prepare variable sizes needed by xmacro
        mjModel* m = model;
        int nv = m->nv, nemax = m->nemax, njmax = m->njmax;

        // find field, set
        if( !strcmp(field, "time") )
            setOneField(ptr, sz, &data->time, 1, 1);
        else if( !strcmp(field, "energy") )
            setOneField(ptr, sz, data->energy, 1, 2);

        #define X(type, name, nr, nc) \
        else if( !strcmp(field, #name) && !strcmp("mjtNum", #type) )\
            setOneField(ptr, sz, data->name, model->nr, nc);

            MJDATA_POINTERS

        #undef X
        else
        {
            strcat(field, ": no such field or field cannot be set");
            mexErrMsgTxt(field);
        }
    }
}


// get model sizes
mxArray* size(void)
{
    // compute size
    int size = getnint();

    // create array of names
    const char* name[100] = {
        #define X(name) #name,
            MJMODEL_INTS
        #undef X
    };

    // create option structure
    mxArray* out = mxCreateStructMatrix(1, 1, size, name);

    // set fields
    #define X(name)\
        mxSetField(out, 0, #name, mxCreateDoubleScalar(model->name));
        MJMODEL_INTS
    #undef X

    return out;
}



//------------------------- Jacobians ---------------------------------------------------

static int _round(double x)
{
    double lower = floor(x);
    double upper = ceil(x);

    if( x-lower < upper-x )
        return (int)lower;
    else
        return (int)upper;
}


static void _checkIndex(const double* input, int size, int maxind)
{
    char errmsg[200];

    for( int i=0; i<size; i++ )
    {
        int ind = _round(input[i]);
        if( ind<0 || ind>=maxind )
        {
            sprintf(errmsg, "invalid index %d, should be between 0 and %d", ind, maxind-1);
            mexErrMsgTxt(errmsg);
        }
    }
}


static void getJacobian(int nout, mxArray* pout[], const mxArray* in, const mxArray* in1,
                        const char* command)
{
    const char* typenames[6] = { "body", "bodycom", "site", "geom", "point", "axis" };

    // find object type
    int type;
    for( type=0; type<6; type++ )
        if( !strcmp(command, typenames[type]) )
            break;
    if( type>=6 )
        mexErrMsgTxt("cannot compute Jacobian of this object type");

    // check input data
    if( type<4 )
        checkNumeric(in, "object id list", -1, -1);
    else
        checkNumeric(in, "3D vector", 3, 1);

    // allocate space for one Jacobian in mjData stack
    int nv = model->nv;
    int _mark = data->pstack;
    mjtNum* JP = mj_stackAlloc(data, 3*nv);
    mjtNum* JR = mj_stackAlloc(data, 3*nv);

    // get input data
    const double* input =  mxGetPr(in);
    const mwSize* sz = mxGetDimensions(in);
    const int size = (int)sz[0] * (int)sz[1];

    // object list
    if( type<4 )
    {
        // check indices
        int num = (type<2 ? model->nbody : (type==2 ? model->nsite : model->ngeom));
        _checkIndex(input, size, num);

        // create outputs for translation and rotation Jacobians, get data pointers
        pout[0] = mxCreateDoubleMatrix(3*size, nv, mxREAL);
        double* outP = mxGetPr(pout[0]);
        double* outR = 0;
        if( nout>1 )
        {
            pout[1] = mxCreateDoubleMatrix(3*size, nv, mxREAL);
            outR = mxGetPr(pout[1]);
        }

        // get and copy Jacobians
        for( int i=0; i<size; i++ )
        {
            // call appropriate jacobian function
            if( type==0 )
                mj_jacBody(model, data, JP, JR, _round(input[i]));
            else if( type==1 )
                mj_jacBodyCom(model, data, JP, JR, _round(input[i]));
            else if( type==2 )
                mj_jacSite(model, data, JP, JR, _round(input[i]));
            else
                mj_jacGeom(model, data, JP, JR, _round(input[i]));

            // copy transposed data to i-th position in output matrices
            for( int r=0; r<3; r++ )
                for( int c=0; c<nv; c++ )
                {
                    outP[3*i+r + c*3*size] = (double) JP[r*nv + c];
                    if( outR )
                        outR[3*i+r + c*3*size] = (double) JR[r*nv + c];
                }
        }
    }

    // point or axis
    else
    {
        // get point or axis
        mjtNum vec[3] = {input[0], input[1], input[2]};

        // get body index from second input
        checkNumeric(in1, "body index", 1, 1);
        int ind = _round(*mxGetPr(in1));
        if( ind<0 || ind>=model->nbody )
        {
            char errmsg[200];
            sprintf(errmsg, "invalid body index %d, should be between 0 and %d",
                ind, model->nbody-1);
            mexErrMsgTxt(errmsg);
        }

        // create outputs
        pout[0] = mxCreateDoubleMatrix(3, nv, mxREAL);
        double* outP = mxGetPr(pout[0]);
        double* outR = 0;
        if( nout>1 && type<5 )
        {
            pout[1] = mxCreateDoubleMatrix(3, nv, mxREAL);
            outR = mxGetPr(pout[1]);
        }

        // point: translation and rotation jacobians
        if( type==4 )
        {
            mj_jac(model, data, JP, JR, vec, ind);

            // copy transposed Jacobian
            mjc2mat(outP, 3, nv, JP);
            if( outR )
                mjc2mat(outR, 3, nv, JR);
        }

        // axis: only one jacobian
        {
            // normalize axis
            if( mju_normalize(vec,3)<mjMINVAL )
                mexErrMsgTxt("axis length is too small");

            // get axis jacobian (point is irrelevant)
            mjtNum dummy[3] = {0,0,0};
            mj_jacPointAxis(model, data, 0, JP, dummy, vec, ind);

            mjc2mat(outP, 3, nv, JP);
        }
    }

    // free stack
    data->pstack = _mark;
}



//------------------------- add and diff (with quaternions) ------------------------


// y = mj('add', x, v)  :  y = x + v
mxArray* add(const mxArray* px, const mxArray* pv)
{
    // get mujoco sizes
    int nqpos   = model->nq;
    int ndof    = model->nv;

    // get input sizes
    const mwSize *dimx  = mxGetDimensions(px),        *dimv = mxGetDimensions(pv);
    mwSize        ndx   = mxGetNumberOfDimensions(px), ndv  = mxGetNumberOfDimensions(pv),
                  numx  = mxGetNumberOfElements(px),   numv = mxGetNumberOfElements(pv);
    int nx   = (int)dimx[0];
    int nv   = (int)dimv[0];
    int nVec = (int)numx/nx;    // total number of vectors

    // check dimensions
    if ( ndx != ndv )           mexErrMsgTxt("number of dimensions must match");
    if ( nx-nv != nqpos-ndof )  mexErrMsgTxt("required: size(x,1)-size(v,1)==model.nq-model.nv");
    if ( nx < nqpos )           mexErrMsgTxt("at least nq rows required");
    if ( nVec != numv/nv )      mexErrMsgTxt("all higher dimensions must match");

    // create output
    mxArray* py = mxCreateNumericArray(ndx, dimx, mxDOUBLE_CLASS, mxREAL);

    // allocate space, get pointers
    int _mark = data->pstack;
    mjtNum* qpos = mj_stackAlloc(data, nx);
    mjtNum* qvel = mj_stackAlloc(data, ndof*sizeof(mjtNum));
    mjtNum *x = mxGetPr(px), *v = mxGetPr(pv), *y = mxGetPr(py);

    // loop over vectors: convert to mjtNum, compute, convert back to double
    for( int i=0; i<nVec; i++ )
    {
        mat2mjc(qpos, nqpos, 1, x + i*nx);
        mat2mjc(qvel, ndof,  1, v + i*nv);
        mj_integratePos(model, qpos, qvel, 1);
        mju_add(qpos+nqpos, x + i*nx + nqpos, v + i*nv + ndof, (int)nx - nqpos);
        mjc2mat(y + i*nx, nx, 1, qpos);
    }

    // free stack
    data->pstack = _mark;

    return py;
}


// v = mj('diff', y, x)  :  v = y - x
mxArray* diff(const mxArray* py, const mxArray* px)
{
    // get mujoco sizes
    int nqpos   = model->nq;
    int ndof    = model->nv;

    // get input sizes
    const mwSize *dimx  = mxGetDimensions(px);
    mwSize        ndx   = mxGetNumberOfDimensions(px), ndy  = mxGetNumberOfDimensions(py),
                  numx  = mxGetNumberOfElements(px),   numy = mxGetNumberOfElements(py);
    int nx   = (int)mxGetM(px);
    int nv   = nx - nqpos + ndof;
    int nVec = (int)numx/nx;    // total number of vectors

    // check sizes
    if ( ndx != ndy )       mexErrMsgTxt("number of dimensions must match");
    if ( nx < nqpos )       mexErrMsgTxt("at least nq rows required");
    if ( nx != mxGetM(py) ) mexErrMsgTxt("first dimension must match");
    if ( numx != numy )     mexErrMsgTxt("higher dimensions must match");

    // create output
    mwSize dimv[10];            // dimensions of output
    dimv[0] = nv;               // set the first dimension of dimv
    for( int i=1; i<ndx; i++ )  // copy the rest
        dimv[i]=dimx[i];
    mxArray* pv = mxCreateNumericArray(ndx, dimv, mxDOUBLE_CLASS, mxREAL);

    // allocate space, get pointers
    int _mark = data->pstack;
    mjtNum* qposx = mj_stackAlloc(data, nqpos);
    mjtNum* qposy = mj_stackAlloc(data, nqpos);
    mjtNum* qvel  = mj_stackAlloc(data, nv);
    mjtNum *y = mxGetPr(py), *x = mxGetPr(px), *v = mxGetPr(pv);

    // loop over vectors: convert to mjtNum, compute, convert back to double
    for( int i=0; i<nVec; i++ )
    {
        mat2mjc(qposx, nqpos, 1, x + i*nx);
        mat2mjc(qposy, nqpos, 1, y + i*nx);
        mj_differentiatePos(model, qvel, 1, qposx, qposy); // differencePos has reverse input order
        mju_sub(qvel + ndof, y + i*nx + nqpos, x + i*nx + nqpos, (int)nx - nqpos);
        mjc2mat(v + i*nv, nv, 1, qvel);
    }

    // free stack
    data->pstack = _mark;

    return pv;
}


//------------------------- vectorized, multithreaded 'step' ------------------------------------------

// mj step;                             % regular 'step'
// x_next = mj('step', x, u);           % vectorized step
void mex_step(const int nout, mxArray* pout[], const int nin, const mxArray* pin[])
{
    // standard internal data step
    if (nin==1)
    {
        mj_step(model, data);
        return;
    }
    
    // check nin, nout, precision
    if (nin<3) 
        mexErrMsgTxt("at least 2 inputs needed: x_next=mj('step', x, u)");  
    if (nout>1) 
        mexErrMsgTxt("too many outputs: x_next=mj('step', x, u)");   
    #ifndef mjUSEDOUBLE
        mexErrMsgTxt("vectorized 'step' not yet supported for floats");   
    #endif
    
    // size shortcuts
    int nq = model->nq;
    int nv = model->nv;
    int na = model->na;
    int nx = nq + nv + na;
    int nu = model->nu;    
    
    // 1st input x
    const mwSize* sz = mxGetDimensions(pin[1]);
    if (sz[0] != nx)
        mexErrMsgTxt("size(x,1) ~= nq+nv+na");
    size_t N = mxGetNumberOfElements(pin[1]) / nx;

    // 2nd input u
    const mwSize* sz2 = mxGetDimensions(pin[2]);
    if (sz2[0] != nu)
        mexErrMsgTxt("size(u,1) ~= nu");
    if (mxGetNumberOfElements(pin[2]) / nu != N)
        mexErrMsgTxt("size(x,2) ~= size(u,2)");
    double *px = mxGetPr(pin[1]);
    double *pu = mxGetPr(pin[2]);

    // allocate output
    pout[0] = mxCreateNumericMatrix(nx, N, mxDOUBLE_CLASS, mxREAL);
    double *pxnew = mxGetPr(pout[0]);
    
    // prepare thread schedule  
    int blksz, extra, schedule[NTHREAD][2];    
    blksz = (int)N / NTHREAD;
    extra = (int)N - blksz*NTHREAD;
    for (int i=0; i<NTHREAD; i++)
    {
        schedule[i][0] = (i ? schedule[i - 1][1] : 0);
        schedule[i][1] = schedule[i][0] + blksz + (i<extra);
    }

    // main computation
    #pragma omp parallel for schedule(static) num_threads(NTHREAD)
    for (int i=0; i<NTHREAD; i++)
        for (int j=schedule[i][0]; j<schedule[i][1]; j++)
        {
            // copy state and control into mjData
            mju_copy(DATA[i]->qpos, px + j*nx, nq);
            mju_copy(DATA[i]->qvel, px + j*nx + nq, nv);
            mju_copy(DATA[i]->act,  px + j*nx + nq + nv, na);
            mju_copy(DATA[i]->ctrl, pu + j*nu, nu);

            // call step
            mj_step(model, DATA[i]);
            
            // copy new state into Matlab
            mju_copy(pxnew + j*nx,           DATA[i]->qpos, nq);
            mju_copy(pxnew + j*nx + nq,      DATA[i]->qvel, nv);
            mju_copy(pxnew + j*nx + nq + nv, DATA[i]->act, na);
        }
}


//------------------------- names -------------------------------------------------------

// get name given type and id
mxArray* getName(const mxArray* pin1, const mxArray* pin2)
{
    char nametype[100];
    checkNumeric(pin2, "object id", 1, 1);

    // get name and id
    mxGetString(pin1, nametype, 100);
    int idobj = _round(*mxGetPr(pin2));
    mjtObj idtype = (mjtObj) mju_str2Type(nametype);

    // get string name and return
    return mxCreateString( mj_id2name(model, idtype, idobj) );
}



// get id given type and name
mxArray* getId(const mxArray* pin1, const mxArray* pin2)
{
    if( mxGetClassID(pin2)!=mxCHAR_CLASS )
        mexErrMsgTxt("last argument must be object name");

    char nametype[100], nameobj[100];

    // get names
    mxGetString(pin1, nametype, 100);
    mxGetString(pin2, nameobj, 100);
    mjtObj idtype = (mjtObj) mju_str2Type(nametype);

    // get id and return
    return mxCreateDoubleScalar( mj_name2id(model, idtype, nameobj) );
}



//------------------------- load file ---------------------------------------------------

// declare continuation of load function
void loadFileCont();

void loadFile(const mxArray* pin)
{
    // get filename
    char filename[100];
    mxGetString(pin, filename, 100);

    // close visualization
    visualize_close();

    // delete model and data
    mj_deleteModel(model);
    mj_deleteData(data);
    model = NULL;
    data = NULL;
    for (int i=0; i<NTHREAD; i++)
    {
        mj_deleteData(DATA[i]);
        DATA[i] = NULL;
    }

    // not .mjb extension: convert
    int len = (int) strlen(filename);
    if( len<4 || _strcmpi(filename+len-4, ".mjb") )
    {
        char errmsg[300];
        model = mj_loadXML(filename, errmsg);
        if( errmsg[0] )
            mexPrintf("%s\n",errmsg);
    }
    else
        model = mj_loadModel(filename, 0, 0);
    
    if( !model )
        mexErrMsgTxt("could not load specified file");

    data = mj_makeData(model);
    if( !data )
        mexErrMsgTxt("could not create mjData");        // SHOULD NOT OCCUR

    // allocate mjData copies for multithreading
    for (int i = 0; i<NTHREAD; i++)
        DATA[i] = mj_copyData(0, model, data);

    // make sure all fields are populated
    mj_forward(model, data);
    #ifdef EXTENSIONS
    loadFileCont();
    #endif
}


//------------------------- print schema ---------------------------------------------------
void printSchema()
{
    char schema[20000];
    mj_printSchema(NULL, schema, 20000, 0, 0);
    mexPrintf(schema);
}

//------------------------- main mex API ------------------------------------------------

// declare continuation of mex function
void mexFunctionCont(int nout, mxArray* pout[], int nin, const mxArray* pin[],
                     const char* command);

// declare continuation of exit function
void exitContTop();

// exit function
void exitFunction(void)
{
    #ifdef EXTENSIONS
    exitContTop();
    #endif

    // close visualization
    visualize_close();
    
    // delete model and data
    mj_deleteModel(model);
    mj_deleteData(data);
    model = NULL;
    data = NULL;
    for (int i = 0; i<NTHREAD; i++)
    {
        mj_deleteData(DATA[i]);
        DATA[i] = NULL;
    }
    
    // unlock mex so MATLAB can remove it from memory
    mexUnlock();
    if(mexIsLocked())
        mexPrintf("Mex unlock failed!\n");
    initialized = false;
}


// entry point
void mexFunction(int nout, mxArray* pout[], int nin, const mxArray* pin[])
{
	const int filename_len=500;
    char filename[filename_len], command[100];
	static bool MuJoCoLicenseActivated = true;
    // register exit function only once
    if( !initialized )
    {         
     
        // set MATLAB error handlers
        mju_user_error = mju_MATLAB_error;
        mju_user_warning = mju_MATLAB_warning;
        mju_user_malloc = mju_MATLAB_malloc;
        mju_user_free = mju_MATLAB_free;
        mexAtExit(exitFunction);
        mexLock();
        initialized = true;
    }

    // no inputs': print help, return
    if( !nin )
    {
        mexPrintf(_mj_help);
        return;
    }

    // get command string
    if( mxGetClassID(pin[0])!=mxCHAR_CLASS )
        mexErrMsgTxt("arg 1 must be string command");
    mxGetString(pin[0], command, 100);

    //---------------------------- version
    if( !strcmp(command, "version") )
    {
        if (!nout)
            mexPrintf("MuJoCo version %g\n", mj_version());
        else
            pout[0] = mxCreateDoubleScalar(mj_version());
    }

	//---------------------------- schema
    else if( !strcmp(command, "schema") )
        printSchema();

	//---------------------------- activation
	else if( !strcmp(command, "activate") )
	{
		if( nin!=2 || mxGetClassID(pin[1])!=mxCHAR_CLASS )
            mexErrMsgTxt("MuJoCo license file path expected as input");
		mxGetString(pin[1], filename, filename_len);

        //if(!mj_activate(filename))
        if( mj_license("MUJOCO_LICENSE.TXT")!=mjLICENSE_OK )
			mexErrMsgTxt("License activation error");
		else {
			mexPrintf("License activated\n");
			MuJoCoLicenseActivated = true;
		}
	}

	//---------------------------- deactivation
	else if( !strcmp(command, "deactivate") )
	{   //mj_deactivate();
		mexPrintf("License deactivated\n");
		MuJoCoLicenseActivated = false;
	}


	//---------------------------- THE REST REQUIRE MUJOCO LICENSE
	else if(!MuJoCoLicenseActivated)
		mexErrMsgTxt("MuJoCo license activation required");

	//---------------------------- clear
    else if( !strcmp(command, "clear") )
        exitFunction();

    //---------------------------- load
    else if( !strcmp(command, "load") )
    {
        // check file name
        if( nin!=2 || mxGetClassID(pin[1])!=mxCHAR_CLASS )
            mexErrMsgTxt("single string argument expected");

        loadFile(pin[1]);
    }

    //---------------------------- ismodel
    else if( !strcmp(command, "ismodel") )
        pout[0] = mxCreateDoubleScalar(model!=0);


	//---------------------------- THE REST REQUIRE MODEL TO BE LOADED
    else if( !model )
        mexErrMsgTxt("model has not been loaded");

    //---------------------------- get fields of mjData
    else if( !strcmp(command, "get") )
        getField(nout, pout, nin, pin);

    //---------------------------- set fields of mjData
    else if( !strcmp(command, "set") )
        setDataField(nin, pin);    

    //---------------------------- set fields of mjModel
    else if( !strcmp(command, "setmodelfield") )
        setModelField(nout, pout, nin, pin);    
    
    //---------------------------- size
    else if( !strcmp(command, "size") )
        pout[0] = size();    

    //---------------------------- getname
    else if( !strcmp(command, "getname") )
    {
        if( nin!=3 ||
            mxGetClassID(pin[1])!=mxCHAR_CLASS )
            mexErrMsgTxt("object type and id expected");

        pout[0] = getName(pin[1], pin[2]);
    }

    //---------------------------- getid
    else if( !strcmp(command, "getid") )
    {
        if( nin!=3 ||
            mxGetClassID(pin[1])!=mxCHAR_CLASS ||
            mxGetClassID(pin[2])!=mxCHAR_CLASS )
            mexErrMsgTxt("object type and name expected");

        pout[0] = getId(pin[1], pin[2]);
    }

    //---------------------------- getoption
    else if( !strcmp(command, "getoption") )
        pout[0] = getOption();

    //---------------------------- getmodel
    else if( !strcmp(command, "getmodel") )
        pout[0] = getModel();

    //---------------------------- getmodel
    else if( !strcmp(command, "getdata") )
        pout[0] = getData();

    //---------------------------- getmesh
    else if( !strcmp(command, "getmesh") )
        pout[0] = getMesh();

    //---------------------------- setoption
    else if( !strcmp(command, "setoption") )
    {
        if( nin!=2 || mxGetClassID(pin[1])!=mxSTRUCT_CLASS)
            mexErrMsgTxt("single struct argument expected");
        setOption(pin[1]);
    }

    //---------------------------- setmodel
    else if( !strcmp(command, "setmodel") )
    {
        if( nin!=2 || mxGetClassID(pin[1])!=mxSTRUCT_CLASS)
            mexErrMsgTxt("single struct argument expected");
        setModel(pin[1]);
    }

    //---------------------------- printmodel
    else if (!strcmp(command, "printmodel"))
    {
        if ( nin!=2 || mxGetClassID(pin[1])!=mxCHAR_CLASS )
            mexErrMsgTxt("expected filename as 2nd input");
        else
        {
            mxGetString(pin[1], filename, filename_len);
            mj_printModel(model, filename);
        }
    }

    //---------------------------- printdata
    else if( !strcmp(command, "printdata") )
    {
        if ( nin!=2 || mxGetClassID(pin[1])!=mxCHAR_CLASS ) 
            mexErrMsgTxt("expected filename as 2nd input");
        else
        {
            mxGetString(pin[1], filename, filename_len);
            mj_printData(model, data, filename);
        }
    }

    //---------------------------- save
    else if( !strcmp(command, "save") )
    {
        if( nin!=2 || mxGetClassID(pin[1])!=mxCHAR_CLASS)
            mexErrMsgTxt("filename expected");

        // save model to binary file
        mxGetString(pin[1], filename, filename_len);
        mj_saveModel(model, filename, 0, 0);
    }

    //---------------------------- addpos
    else if( !strcmp(command, "add") )
    {
        if( nin!=3 )
            mexErrMsgTxt("two arguments expected: x and dx");

        pout[0] = add(pin[1], pin[2]);
    }

    //---------------------------- difpos
    else if( !strcmp(command, "diff") )
    {
        if( nin!=3 )
            mexErrMsgTxt("two arguments expected: qpos1 and qpos2");

        pout[0] = diff(pin[1], pin[2]);
    }

    //---------------------------- jacobian
    else if( command[0]=='j' && command[1]=='a'&& command[2]=='c' )
    {
        if( mxGetClassID(pin[1])!=mxDOUBLE_CLASS )
            mexErrMsgTxt("numeric arguments expected");

        if( nin>2 )
            getJacobian(nout, pout, pin[1], pin[2], command+3);
        else
            getJacobian(nout, pout, pin[1], 0, command+3);
    }

    //---------------------------- reset
    else if( !strcmp(command, "reset") )
        mj_resetData(model, data);

    //---------------------------- forward
    else if( !strcmp(command, "forward") )
        mj_forward(model, data);

    //---------------------------- integrate
    else if( !strcmp(command, "integrate") )
        mj_Euler(model, data);

    //---------------------------- inverse
    else if( !strcmp(command, "inverse") )
        mj_inverse(model, data);
    
    //---------------------------- step
    else if (!strcmp(command, "step"))
        mex_step(nout, pout, nin, pin);

    //---------------------------- step1
    else if( !strcmp(command, "step1") )
        mj_step1(model, data);

    //---------------------------- step2
    else if( !strcmp(command, "step2") )
        mj_step2(model, data);

    //---------------------------- kinematics
    else if( !strcmp(command, "kinematics") )
        mj_fwdPosition(model, data);

    //---------------------------- plot
    else if (!strcmp(command, "plot")){
        visualize_init();
    }

    //---------------------------- continuation from matlab_functions.cpp
    #ifdef EXTENSIONS
    else
        mexFunctionCont(nout, pout, nin, pin, command);
    #else
    else
        mexErrMsgTxt("command not recognized, type 'mj' for help");    
    #endif

}
