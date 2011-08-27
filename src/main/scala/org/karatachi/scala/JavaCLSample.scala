package org.karatachi.scala

import scala.math._

import org.bridj._

import com.nativelibs4java.opencl._
import com.nativelibs4java.opencl.util._
import com.nativelibs4java.opencl.CLMem.Usage
import com.nativelibs4java.util._

object JavaCLSample extends App {
  val context = JavaCL.createBestContext
  val queue = context.createDefaultQueue()

  val n = 1024

  val aPtr = Pointer.allocateFloats(n)
  val bPtr = Pointer.allocateFloats(n)

  (0 until n).foreach { i =>
    aPtr.set(i, cos(i).toFloat)
    bPtr.set(i, sin(i).toFloat)
  }

  val a = context.createFloatBuffer(Usage.Input, aPtr)
  val b = context.createFloatBuffer(Usage.Input, bPtr)

  val out = context.createFloatBuffer(Usage.Output, n)

  val program = context.createProgram("""
__kernel void add_floats(__global const float* a, __global const float* b, __global float* out, int n) {
    int i = get_global_id(0);
    if (i >= n)
        return;

    out[i] = a[i] + b[i];
}
""")

  val kernel = program.createKernel("add_floats")
  kernel.setArgs(a, b, out, n.asInstanceOf[AnyRef])
  val event = kernel.enqueueNDRange(queue, Array(n))

  val outPtr = out.read(queue, event)

  (0 until 10).foreach { i =>
    println("out[%d] = %f".format(i, outPtr.get(i)))
  }
}
