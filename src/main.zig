const std = @import("std");
const tinystl = @import("./lib.zig");

pub fn main() !void {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    // var allocator = gpa.allocator();
    var allocator = std.heap.page_allocator;
    // defer _ = allocator.deinit();

    var arg_iter = try std.process.argsWithAllocator(allocator);
    _ = arg_iter.skip();
    const in_path = arg_iter.next();
    var in_file = try std.fs.cwd().openFile(in_path.?, .{});

    var data = try tinystl.StlData.readFromFile(in_file, &allocator);
    defer data.deinit();

    var stdout = std.io.getStdOut().writer();
    try stdout.print("{}\n", .{data.triangles.items.len});
    // const out_file = arg_iter.next();
    // try data.writeAsciiFile(out_file.?);
}
