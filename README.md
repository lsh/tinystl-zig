 # TinySTL - A small loader for STL files.
 This project is heavily inspired by, and adapted from, [cry-inc's microstl library](https://github.com/cry-inc/microstl).
 The goal is to provide a zero-dependency way to easily load and write STL files.
 It is assumed that all binary files are little endian.

 # Example
 ```zig
const StlData = @include("stldata.zig").StlData;

pub fn main() !void {
  var allocator = std.heap.page_allocator;
  var file = std.fs.cwd().openFile("my_mesh.stl", .{});
  defer file.close();
  var data = StlData.readFromFile(&file, &allocator, .{});
  defer data.deinit();

  var out_file = std.fs.cwd().createFile("my_mesh_output.stl", .{});
  defer out_file.close();
  data.writeBinaryFile(&file, .{});
}
 ```
