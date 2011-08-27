package org.karatachi.scala

import scala.math._

import org.bridj._

import com.nativelibs4java.opencl._
import com.nativelibs4java.opencl.util._
import com.nativelibs4java.opencl.CLMem.Usage
import com.nativelibs4java.util._

object DataParallel extends App {
  val context = JavaCL.createBestContext
  val queue = context.createDefaultQueue()

  val aPtr = Pointer.allocateFloats(16)
  val bPtr = Pointer.allocateFloats(16)

  for (i <- (0 until 4); j <- (0 until 4)) {
    aPtr.set(i * 4 + j, i * 4 + j + 1)
    bPtr.set(i * 4 + j, j * 4 + i + 1)
  }

  val a = context.createFloatBuffer(Usage.Input, aPtr)
  val b = context.createFloatBuffer(Usage.Input, bPtr)

  val out = context.createFloatBuffer(Usage.Output, 16)

  val program = context.createProgram("""
__kernel void dataParallel(__global const float4* a, __global const float4* b, __global float4* c) {
    int i = get_global_id(0);
    c[i].xyz = a[i].xyz + b[i].xyz;
    c[i].w   = a[i].w   * b[i].w;
}
""")

  val kernel = program.createKernel("dataParallel")
  kernel.setArgs(a, b, out)
  val event = kernel.enqueueNDRange(queue, Array(4))

  val outPtr = out.read(queue, event)

  val array = outPtr.getFloats()
  for (i <- (0 until 4)) {
    for (j <- (0 until 4)) {
      print("%7.2f".format(array(i * 4 + j)))
    }
    println()
  }
}
