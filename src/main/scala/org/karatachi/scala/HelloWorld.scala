package org.karatachi.scala

import scala.math._

import org.bridj._

import com.nativelibs4java.opencl._
import com.nativelibs4java.opencl.util._
import com.nativelibs4java.opencl.CLMem.Usage
import com.nativelibs4java.util._

object HelloWorld extends App {
  val context = JavaCL.createBestContext
  val queue = context.createDefaultQueue()

  val out = context.createByteBuffer(Usage.Output, 16)

  val program = context.createProgram("""
__kernel void hello(__global char* string) {
    string[0] = 'H';
    string[1] = 'e';
    string[2] = 'l';
    string[3] = 'l';
    string[4] = 'o';
    string[5] = ',';
    string[6] = ' ';
    string[7] = 'W';
    string[8] = 'o';
    string[9] = 'r';
    string[10] = 'l';
    string[11] = 'd';
    string[12] = '!';
    string[13] = '\0';
}
""")

  val kernel = program.createKernel("hello")
  kernel.setArgs(out)
  val event = kernel.enqueueTask(queue)

  val outPtr = out.read(queue, event)

  println(new String(outPtr.getBytes(), "UTF-8"))
}
