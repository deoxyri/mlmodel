import pyopencl
from pyopencl.tools import get_test_platforms_and_devices
import pyopencl as cl
import numpy as np
import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'
print(get_test_platforms_and_devices())
# ----------------------------------------------------------------------------------------------------------------------
# RANDOM DATA GENERATION
(n, m, p) = (3, 4, 5)

# a = np.random.randn(n, m).astype(np.float32)
# b = np.random.randn(m, p).astype(np.float32)
a = np.random.randint(2, size=(n * m))
b = np.random.randint(2, size=(m * p))
c = np.zeros((n * p), dtype=np.float32)

a = a.astype(np.float32)
b = b.astype(np.float32)
# ----------------------------------------------------------------------------------------------------------------------
# setup of the context, queue, and buffers
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
# ----------------------------------------------------------------------------------------------------------------------
# kernel is defined
prg = cl.Program(ctx, """__kernel void multiply(ushort n,ushort m, ushort p, __global float *a,
    __global float *b, __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = 0.0f;
      int rowC = gid/p;
      int colC = gid%p;
      __global float *pA = &a[rowC*m];
      __global float *pB = &b[colC];
      for(int k=0; k<m; k++)
      {
         pB = &b[colC+k*p];
         c[gid] += (*(pA++))*(*pB);
      }
    }
    """).build()
# ----------------------------------------------------------------------------------------------------------------------
# main function to execute the OpenCL code
prg.multiply(queue, c.shape, None, np.uint16(n), np.uint16(m), np.uint16(p), a_buf, b_buf, c_buf)
a_mul_b = np.empty_like(c)
cl.enqueue_copy(queue, a_mul_b, c_buf)
# ----------------------------------------------------------------------------------------------------------------------
print("matrix A:")
print(a.reshape(n, m))
print("matrix B:")
print(b.reshape(m, p))
print("multiplied A*B:")
print(a_mul_b.reshape(n, p))
