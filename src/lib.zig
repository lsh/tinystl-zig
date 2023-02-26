const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

// All based on the code from microstl.h - STL file format parser
// https://github.com/cry-inc/microstl/blob/master/include/microstl.h

// # TinySTL - A small loader for STL files.
// This project is heavily inspired by, and adapted from, [cry-inc's microstl library](https://github.com/cry-inc/microstl).
// The goal is to provide a zero-dependency way to easily load and write STL files.
// It is assumed that all binary files are little endian.
//
//  # Example
//  ```zig
// const StlData = @include("stldata.zig").StlData;

// pub fn main() !void {
//   var allocator = std.heap.page_allocator;
//   var file = try std.fs.cwd().openFile("my_mesh.stl", .{});
//   defer file.close();
//   var data = try StlData.readFromFile(&file, &allocator, .{});
//   defer data.deinit();

//   var out_file = try std.fs.cwd().createFile("my_mesh_output.stl", .{});
//   defer out_file.close();
//   try data.writeBinaryFile(&file, .{});
// }
//  ```

/// In binary STL files, a triangle will always consist of 50 bytes.
/// A triangle consists of:
///
/// ```text
/// Normal: [f32; 3] - 12 bytes
/// Vertex 1: [f32; 3] - 12 bytes
/// Vertex 2: [f32; 3] - 12 bytes
/// Vertex 3: [f32; 3] - 12 bytes
/// Attirbute byte count: [u8; 2] - 2 bytes (generally {0, 0})
/// ```
/// For more information see the [Wikipedia page][https://en.wikipedia.org/wiki/STL_(file_format)#Binary_STL] on the format.
const facet_binary_size = 50;

const f32x3_size = @sizeOf(f32) * 3;

// From https://github.com/cry-inc/microstl/blob/ec3868a14d8eff40f7945b39758edf623f609b6f/include/microstl.h#L177
fn stringTrim(input: []const u8) []const u8 {
    var index: usize = 0;
    var input_size = input.len;
    while (index < input_size and std.ascii.isWhitespace(input[index])) : (index += 1) {}

    var rest = input[index..];
    if (rest.len == 0) {
        return rest;
    }

    index = rest.len - 1;
    while (std.ascii.isWhitespace(rest[index])) : (index -= 1) {
        if (index == 0) {
            break;
        }
    }

    return rest[0..(index + 1)];
}

// The binary STL format contains an 80 byte header.
// It should *never* start with `b"solid"`.
// For more information see the [Wikipedia page][https://en.wikipedia.org/wiki/STL_(file_format)#Binary_STL] on the format.
const header_binary_size = 80;

const Vec3 = packed struct {
    x: f32 = 0.0,
    y: f32 = 0.0,
    z: f32 = 0.0,

    pub fn splat(v: f32) Vec3 {
        return Vec3{ .x = v, .y = v, .z = v };
    }

    pub fn init(x: f32, y: f32, z: f32) Vec3 {
        return Vec3{
            .x = x,
            .y = y,
            .z = z,
        };
    }

    pub fn dot(self: Vec3, other: Vec3) f32 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    pub fn length(self: Vec3) f32 {
        return std.math.sqrt(self.dot(self));
    }

    pub fn fromf32x3(buffer: [3]f32) Vec3 {
        return Vec3{ .x = buffer[0], .y = buffer[1], .z = buffer[2] };
    }

    /// Converts a triple such as `1.0 1.0 1.0` into a Vec3.
    pub fn fromString(buffer: []const u8) Error!Vec3 {
        var tokenizer = std.mem.tokenize(u8, buffer, " ");
        return Vec3.fromTokenIter(&tokenizer);
    }

    /// Converts a triple such as `1.0 1.0 1.0` into a Vec3.
    pub fn fromTokenIter(iter: *std.mem.TokenIterator(u8)) Error!Vec3 {
        var result = [3]f32{ undefined, undefined, undefined };
        var i: usize = 0;
        while (iter.*.next()) |token| {
            std.debug.assert(i < 3);
            result[i] = std.fmt.parseFloat(f32, token) catch {
                return Error.Parse;
            };
            i += 1;
        }

        if (i != 3) {
            return Error.Parse;
        }

        return fromf32x3(result);
    }
};

test "Vec3 From Str" {
    const res = try Vec3.fromString("1.0 2.0 3.0");
    const expected = Vec3.init(1.0, 2.0, 3.0);
    std.debug.assert(res.x == expected.x and res.y == expected.y and res.z == expected.z);
}

test "Vec3 From Str whitespace" {
    const res = try Vec3.fromString("   1.0    2.0      3.0");
    const expected = Vec3.init(1.0, 2.0, 3.0);
    std.debug.assert(res.x == expected.x and res.y == expected.y and res.z == expected.z);
}

/// Each facet contains a copy of all three vertex coordinates and a normal
pub const Triangle = packed struct {
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,

    /// Set the facet normal based on the vertices.
    fn calculateNormals(self: Triangle) Vec3 {
        const u = Vec3{
            .x = self.v2.x - self.v1.x,
            .y = self.v2.y - self.v1.y,
            .z = self.v2.z - self.v1.z,
        };
        const v = Vec3{
            .x = self.v3.x - self.v1.x,
            .y = self.v3.y - self.v1.y,
            .z = self.v3.z - self.v1.z,
        };
        var normal = Vec3{ .x = u.y * v.z - u.z * v.y, .y = u.z * v.x - u.x * v.z, .z = u.x * v.y - u.y * v.x };

        const len = normal.length();
        normal.x /= len;
        normal.y /= len;
        normal.z /= len;
        return normal;
    }

    /// Fix normals on the facet beneath a certain epsilon
    fn checkAndFixNormals(self: Triangle, normal: Vec3) Vec3 {
        const normal_length_deviation_limit = 0.001;

        const n = if (normal.x == 0.0 and normal.y == 0.0 and normal.z == 0.0)
            self.calculateNormals()
        else
            normal;

        const len = n.length();
        return if (std.math.fabs(len - 1.0) > normal_length_deviation_limit) {
            return self.calculateNormals();
        } else n;
    }

    pub fn fromBuffer(buffer: []const u8) Triangle {
        std.debug.assert(buffer.len == 36);
        var tri = Triangle{ .v1 = undefined, .v2 = undefined, .v3 = undefined };
        comptime {
            inline for (0..3) |i| {
                const chunk_start = i * f32x3_size;
                const buf = buffer[chunk_start..(chunk_start + f32x3_size)];
                const data = @bitCast(Vec3, buf.*);
                const field = "v" ++ [_]u8{std.fmt.digitToChar(i + 1, std.fmt.Case.lower)};
                @field(tri, field) = data;
            }
        }
        return tri;
    }
};

/// Possible errors that come from loading a file
const Error = error{ MissingData, Unexpected, Parse, TooManyFacets, Read, Write, OutOfMemory, OpenFile };

pub const Encoding = enum { binary, ascii };

pub const StlReadOpts = struct {
    /// Set to true to force recalculatian normals using vertex data.
    /// By default, recalculation is only done for zero normal vectors
    /// or normal vectors with invalid length.
    force_normals: bool = false,
    /// Set to true to disable all reculation of normal vectors.
    /// By default, recalculation is only done for zero normal vectors
    /// or normal vectors with invalid length.
    disable_normals: bool = false,
};

pub const StlWriteOpts = struct {
    /// Set to true to write zero normals in the output.
    nullify_normals: bool = false,
};

/// The container for all STL data.
pub const StlData = struct {
    allocator: *Allocator,
    // data
    triangles: ArrayList(Triangle),
    normals: ArrayList(Vec3),
    header: [header_binary_size]u8,

    name: ?[]u8 = null,

    /// The encoding from which the mesh was read.
    encoding: ?Encoding = null,

    pub fn init(allocator: *Allocator) StlData {
        return StlData{
            .allocator = allocator,
            .triangles = ArrayList(Triangle).init(allocator.*),
            .normals = ArrayList(Vec3).init(allocator.*),
            .header = [_]u8{0} ** 80,
        };
    }

    pub fn deinit(self: *StlData) void {
        self.triangles.deinit();
        self.normals.deinit();
        if (self.name != null) {
            self.allocator.free(self.name.?);
        }
    }

    /// Creates and populates a ``StlData`` from a file path.
    pub fn readFromFile(file: *std.fs.File, allocator: *Allocator, opts: StlReadOpts) Error!StlData {
        var buf_reader = std.io.bufferedReader(file.*.reader());
        var reader = buf_reader.reader();
        return fromReader(&reader, allocator, opts);
    }

    /// Constructs `StlData` from a struct that conforms to the `Reader` interface.
    pub fn fromReader(reader: anytype, allocator: *Allocator, opts: StlReadOpts) Error!StlData {
        var data = StlData.init(allocator);

        // Get the first line to check if ascii or binary
        var peek_stream = std.io.peekStream(5, reader.*);
        var peek_reader = peek_stream.reader();
        var buf: [5]u8 = undefined;
        var n = peek_reader.read(&buf) catch {
            return Error.Read;
        };

        if (n != buf.len) {
            return Error.MissingData;
        }

        peek_stream.putBack(&buf) catch {
            return Error.Read;
        };

        if (std.mem.eql(u8, buf[0..5], "solid")) {
            try data.readAsciiBuffer(&peek_reader, opts);
            data.encoding = Encoding.ascii;
        } else {
            try data.readBinaryBuffer(&peek_reader, opts);
            data.encoding = Encoding.binary;
        }

        return data;
    }
    /// Write the contents of a ``StlData`` to a file using the ASCII specification.
    /// Will truncate the path if it does not exist by default
    pub fn writeAsciiFile(self: *StlData, file: *std.fs.File, opts: StlWriteOpts) Error!void {
        var buf_writer = std.io.bufferedWriter(file.*.writer());
        var writer = buf_writer.writer();
        try self.writeAscii(&writer, opts);
        buf_writer.flush() catch {
            return Error.Write;
        };
    }

    /// Write the contents of a ``StlData`` to a file using the ASCII specification.
    /// Will truncate the path if it does not exist by default
    pub fn writeBinaryFile(self: *StlData, file: *std.fs.File, opts: StlWriteOpts) Error!void {
        var buf_writer = std.io.bufferedWriter(file.*.writer());
        var writer = buf_writer.writer();
        try self.writeBinary(&writer, opts);
        buf_writer.flush() catch {
            return Error.Write;
        };
    }

    /// Reset all data within the reader.
    pub fn clear(self: *StlData) void {
        if (self.name != null) {
            self.allocator.free(self.name.?);
            self.name = null;
        }
        self.triangles.clearRetainingCapacity();
        self.normals.clearRetainingCapacity();
        self.header = [_]u8{0} ** 80;
        self.encoding = null;
    }

    /// For internal use.
    /// Sets the contents ``StlData`` from a binary buffer.
    fn readBinaryBuffer(self: *StlData, reader: anytype, opts: StlReadOpts) Error!void {
        self.clear();
        errdefer self.clear();

        var header_buffer: [header_binary_size]u8 = [_]u8{0} ** header_binary_size;
        const header_read_amount = reader.read(&header_buffer) catch {
            return Error.Read;
        };
        if (header_read_amount != header_binary_size) {
            return Error.MissingData;
        }

        var facet_count_buf = [4]u8{ 0, 0, 0, 0 };
        const facet_count_read_amount = reader.read(&facet_count_buf) catch {
            return Error.Read;
        };
        if (facet_count_read_amount != 4) {
            return Error.MissingData;
        }
        const facet_count = @bitCast(u32, facet_count_buf);
        if (facet_count == 0) {
            return Error.MissingData;
        }

        var buffer: [facet_binary_size]u8 = undefined;
        for (0..facet_count) |_| {
            const buffer_read_amount = reader.read(&buffer) catch {
                return Error.Read;
            };
            if (buffer_read_amount != facet_binary_size) {
                return Error.MissingData;
            }

            const n = @bitCast(Vec3, buffer[0..f32x3_size].*);
            const facet = Triangle.fromBuffer(buffer[f32x3_size..(facet_binary_size - 2)]);

            const normal = if (opts.force_normals and !opts.disable_normals)
                facet.calculateNormals()
            else if (!opts.disable_normals) facet.checkAndFixNormals(n) else n;

            self.normals.append(normal) catch {
                return Error.OutOfMemory;
            };
            self.triangles.append(facet) catch {
                return Error.OutOfMemory;
            };
        }
    }

    /// For internal use.
    /// Sets the contents ``StlData`` from an ASCII buffer.
    fn readAsciiBuffer(self: *StlData, reader: anytype, opts: StlReadOpts) !void {
        self.clear();
        errdefer self.clear();

        // State machine variables
        var active_solid: bool = false;
        var active_facet: bool = false;
        var active_loop: bool = false;
        var solid_count: usize = 0;
        var loop_count: usize = 0;
        var vertex_count: usize = 0;

        var line_number: usize = 1;
        var v = [_]Vec3{Vec3{}} ** 3;
        var n = Vec3{};

        // Line reader with loop to work the state machine
        var buf: [1024]u8 = undefined;
        while (reader.readUntilDelimiterOrEof(&buf, '\n') catch {
            return Error.Read;
        }) |line| {
            var line_tokens_iter = std.mem.tokenize(u8, line, " \t");
            const line_start = line_tokens_iter.next() orelse "";
            if (std.mem.eql(u8, line_start, "solid")) {
                if (active_solid or solid_count != 0) {
                    return Error.Unexpected;
                }
                var mesh_name = stringTrim(line["solid".len..]);
                active_solid = true;
                self.name = self.allocator.alloc(u8, mesh_name.len) catch {
                    return Error.OutOfMemory;
                };
                for (mesh_name, 0..) |c, i| self.name.?[i] = c;
            }
            if (std.mem.eql(u8, line_start, "endsolid")) {
                if (!active_solid or active_facet or active_loop) {
                    return Error.Unexpected;
                }
                active_solid = false;
                solid_count += 1;
            }
            if (std.mem.eql(u8, line_start, "facet")) {
                _ = line_tokens_iter.next(); // skip the word "normal"
                if (!active_solid or active_loop or active_facet) {
                    return Error.Unexpected;
                }
                active_facet = true;

                n = try Vec3.fromTokenIter(&line_tokens_iter);
            }
            if (std.mem.eql(u8, line_start, "endfacet")) {
                if (!active_solid or active_loop or !active_facet or loop_count != 1) {
                    return Error.Unexpected;
                }
                active_facet = false;
                loop_count = 0;
                const facet = Triangle{
                    .v1 = v[0],
                    .v2 = v[1],
                    .v3 = v[2],
                };

                const normal = if (opts.force_normals and !opts.disable_normals)
                    facet.calculateNormals()
                else if (!opts.disable_normals) facet.checkAndFixNormals(n) else n;

                self.triangles.append(facet) catch {
                    return Error.OutOfMemory;
                };
                self.normals.append(normal) catch {
                    return Error.OutOfMemory;
                };
            }
            if (std.mem.eql(u8, line_start, "outer")) {
                if (!active_solid or !active_facet or active_loop) {
                    return Error.Unexpected;
                }
                active_loop = true;
            }
            if (std.mem.eql(u8, line_start, "endloop")) {
                if (!active_solid or !active_facet or !active_loop or vertex_count != 3) {
                    return Error.Unexpected;
                }
                active_loop = false;
                loop_count += 1;
                vertex_count = 0;
            }
            if (std.mem.eql(u8, line_start, "vertex")) {
                if (!active_solid or !active_facet or !active_loop or vertex_count >= 3) {
                    return Error.Unexpected;
                }
                const triplet = try Vec3.fromTokenIter(&line_tokens_iter);
                v[vertex_count] = triplet;
                vertex_count += 1;
            }

            line_number += 1;
        }

        if (active_solid or active_facet or active_loop or solid_count == 0) {
            return Error.MissingData;
        }
    }

    /// Write the contents of a ``StlData`` to a buffer using the ASCII specification.
    pub fn writeAscii(self: *StlData, writer: anytype, opts: StlWriteOpts) Error!void {
        writer.writeAll("solid ") catch {
            return Error.Write;
        };
        if (self.name != null) {
            writer.writeAll(self.name.?) catch {
                return Error.Write;
            };
        }
        writer.writeAll("\n") catch {
            return Error.Write;
        };
        for (self.triangles.items, self.normals.items) |triangle, normal| {
            if (opts.nullify_normals) {
                writer.writeAll("  facet normal 0 0 0\n") catch {
                    return Error.Write;
                };
            } else {
                writer.print("  facet normal {} {} {}\n", .{ normal.x, normal.y, normal.z }) catch {
                    return Error.Write;
                };
            }
            writer.writeAll("    outer loop\n") catch {
                return Error.Write;
            };
            inline for (.{ "v1", "v2", "v3" }) |v| {
                const vertex = @field(triangle, v);
                writer.print("      vertex {} {} {}\n", .{ vertex.x, vertex.y, vertex.z }) catch {
                    return Error.Write;
                };
            }
            writer.writeAll("    endloop\n") catch {
                return Error.Write;
            };
            writer.writeAll("  endfacet\n") catch {
                return Error.Write;
            };
        }
        writer.writeAll("endsolid\n") catch {
            return Error.Write;
        };
    }

    /// Write the contents of a ``StlData`` to a buffer using the binary specification.
    pub fn writeBinary(self: *StlData, writer: anytype, opts: StlWriteOpts) Error!void {
        writer.writeAll(&self.header) catch {
            return Error.Write;
        };

        const n_triangles = @intCast(u32, self.triangles.items.len);
        const n_triangles_bytes = @bitCast([4]u8, n_triangles);
        writer.writeAll(&n_triangles_bytes) catch {
            return Error.Write;
        };
        const null_bytes = [1]u8{0} ** 12;

        for (self.triangles.items, self.normals.items) |triangle, normal| {
            if (opts.nullify_normals) {
                writer.writeAll(&null_bytes) catch {
                    return Error.Write;
                };
            } else {
                const normal_bytes = @bitCast([f32x3_size]u8, normal);
                writer.writeAll(&normal_bytes) catch {
                    return Error.Write;
                };
            }

            inline for (.{ "v1", "v2", "v3" }) |field| {
                const vertex = @field(triangle, field);
                const vertex_bytes = @bitCast([f32x3_size]u8, vertex);
                writer.writeAll(&vertex_bytes) catch {
                    return Error.Write;
                };
            }

            writer.writeAll(&[_]u8{ 0, 0 }) catch {
                return Error.Write;
            };
        }
    }
};

// All based on the tests from microstl
// https://github.com/cry-inc/microstl/blob/master/tests/tests.cpp

test "minimal ascii file" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/simple_ascii.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();
    try testing.expectEqualStrings("minimal", res.name.?);
    try testing.expectEqual(Encoding.ascii, res.encoding.?);
    const empty_slice = [_]u8{0} ** header_binary_size;
    try testing.expectEqualSlices(u8, &empty_slice, &res.header);
    try testing.expect(res.triangles.items.len == 1);
    try testing.expectEqual(Vec3{ .x = -1.0 }, res.normals.items[0]);
    try testing.expectEqual(Vec3{}, res.triangles.items[0].v1);
    try testing.expectEqual(Vec3{ .z = 1.0 }, res.triangles.items[0].v2);
    try testing.expectEqual(Vec3{ .y = 1.0, .z = 1.0 }, res.triangles.items[0].v3);
}

test "ascii file with creative white space" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/crazy_whitespace_ascii.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();
    try testing.expectEqualStrings("min \t imal", res.name.?);
    try testing.expectEqual(Encoding.ascii, res.encoding.?);
    const empty_slice = [_]u8{0} ** header_binary_size;
    try testing.expectEqualSlices(u8, &empty_slice, &res.header);
    try testing.expect(res.triangles.items.len == 1);
    try testing.expectEqual(Vec3{ .x = -1.0 }, res.normals.items[0]);
    try testing.expectEqual(Vec3{}, res.triangles.items[0].v1);
    try testing.expectEqual(Vec3{ .z = 1.0 }, res.triangles.items[0].v2);
    try testing.expectEqual(Vec3{ .y = 1.0, .z = 1.0 }, res.triangles.items[0].v3);
}

test "small ascii file" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/half_donut_ascii.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();
    try testing.expectEqualStrings("Half Donut", res.name.?);
    try testing.expectEqual(Encoding.ascii, res.encoding.?);
    const empty_slice = [_]u8{0} ** header_binary_size;
    try testing.expectEqualSlices(u8, &empty_slice, &res.header);
    try testing.expect(res.triangles.items.len == 288);
}

test "binary file" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/stencil_binary.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();
    try testing.expect(res.name == null);
    try testing.expectEqual(Encoding.binary, res.encoding.?);
    const empty_slice = [_]u8{0} ** header_binary_size;
    try testing.expectEqualSlices(u8, &empty_slice, &res.header);
    try testing.expect(res.triangles.items.len == 2330);
}

test "binary freecad" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/box_freecad_binary.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();
    try testing.expect(res.name == null);
    try testing.expectEqual(Encoding.binary, res.encoding.?);
    try testing.expect(res.triangles.items.len == 12);
    try testing.expectEqual(Vec3{ .z = 1.0 }, res.normals.items[11]);
    try testing.expectEqual(Vec3{ .x = 20.0, .z = 20.0 }, res.triangles.items[11].v1);
    try testing.expectEqual(Vec3{ .z = 20.0 }, res.triangles.items[11].v2);
    try testing.expectEqual(Vec3{ .x = 20.0, .y = -20.0, .z = 20.0 }, res.triangles.items[11].v3);
}

test "meshlab ascii" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/box_meshlab_ascii.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();
    try testing.expectEqualStrings("STL generated by MeshLab", res.name.?);
    try testing.expectEqual(Encoding.ascii, res.encoding.?);
    try testing.expect(res.triangles.items.len == 12);
    try testing.expectEqual(Vec3{ .z = 1.0 }, res.normals.items[11]);
    try testing.expectEqual(Vec3{ .x = 20.0, .z = 20.0 }, res.triangles.items[11].v1);
    try testing.expectEqual(Vec3{ .z = 20.0 }, res.triangles.items[11].v2);
    try testing.expectEqual(Vec3{ .x = 20.0, .y = -20.0, .z = 20.0 }, res.triangles.items[11].v3);
}

test "utf-8 file name" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/简化字.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();
    try testing.expect(res.triangles.items.len == 1);
}

test "sphere" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/sphere_binary.stl", .{});
    defer file.close();

    var reader = file.reader();
    var data = try StlData.fromReader(&reader, &allocator, .{ .force_normals = true });
    defer data.deinit();

    try testing.expect(data.triangles.items.len == 1360);
    const radius = 10.0;
    const allowed_deviation = 0.00001;
    for (data.triangles.items, data.normals.items) |f, normal| {
        const length1 = f.v1.length();
        try testing.expect(std.math.fabs(length1 - radius) < allowed_deviation);
        const length2 = f.v2.length();
        try testing.expect(std.math.fabs(length2 - radius) < allowed_deviation);
        const length3 = f.v3.length();
        try testing.expect(std.math.fabs(length3 - radius) < allowed_deviation);

        // Check if origin is "behind" the normal plane
        // (normal of all sphere surface triangle should point away from the origin)
        const origin = Vec3{};
        const tmp = Vec3{
            .x = origin.x - f.v1.x,
            .y = origin.y - f.v2.y,
            .z = origin.z - f.v2.z,
        };
        const dot = normal.dot(tmp);
        try testing.expect(dot < 0.0);

        // Check normal vector length
        const length = normal.length();
        try testing.expect(std.math.fabs(length - 1.0) < allowed_deviation);
    }
}

test "incomplete vertex" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/incomplete_vertex_ascii.stl", .{});
    defer file.close();

    try testing.expectError(Error.Parse, StlData.readFromFile(&file, &allocator, .{}));
}

test "incomplete normal" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/incomplete_normal_ascii.stl", .{});
    defer file.close();

    try testing.expectError(Error.Parse, StlData.readFromFile(&file, &allocator, .{}));
}

test "empty file" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/empty_file.stl", .{});
    defer file.close();

    try testing.expectError(Error.MissingData, StlData.readFromFile(&file, &allocator, .{}));
}

test "incomplete binary" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/incomplete_binary.stl", .{});
    defer file.close();

    try testing.expectError(Error.MissingData, StlData.readFromFile(&file, &allocator, .{}));
}

test "simple writer" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/box_meshlab_ascii.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var tmp_binary = try tmp.dir.createFile("test_binary.stl", .{});
    defer tmp_binary.close();
    try res.writeBinaryFile(&tmp_binary, .{});

    var tmp_ascii = try tmp.dir.createFile("test_ascii.stl", .{});
    defer tmp_ascii.close();
    try res.writeAsciiFile(&tmp_ascii, .{});
}

test "nulled normals" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/box_meshlab_ascii.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var tmp_binary = try tmp.dir.createFile("test_binary.stl", .{});
    defer tmp_binary.close();
    try res.writeBinaryFile(&tmp_binary, .{ .nullify_normals = true });

    var tmp_ascii = try tmp.dir.createFile("test_ascii.stl", .{});
    defer tmp_ascii.close();
    try res.writeAsciiFile(&tmp_ascii, .{ .nullify_normals = true });
}

test "write buffer" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/box_meshlab_ascii.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();
    var arraylist = ArrayList(u8).init(allocator);
    var writer = arraylist.writer();
    try res.writeBinary(&writer, .{});
    try testing.expect(arraylist.items.len == 80 + 4 + 12 * (12 * 4 + 2));
}

test "full cycle" {
    var allocator = std.heap.page_allocator;
    var file = try std.fs.cwd().openFile("testdata/box_meshlab_ascii.stl", .{});
    defer file.close();

    var res = try StlData.readFromFile(&file, &allocator, .{});
    defer res.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var tmp_binary = try tmp.dir.createFile("test_binary.stl", .{ .read = true });
    try res.writeBinaryFile(&tmp_binary, .{});

    var tmp_ascii = try tmp.dir.createFile("test_ascii.stl", .{ .read = true });
    var tmp_ascii_w = try tmp.dir.openFile("test_ascii.stl", .{ .mode = std.fs.File.OpenMode.read_write });
    try res.writeAsciiFile(&tmp_ascii_w, .{});

    var new_data = try StlData.readFromFile(&tmp_ascii, &allocator, .{});
    defer new_data.deinit();
    try testing.expect(new_data.triangles.items.len == 12);
    try testing.expectEqual(Encoding.ascii, new_data.encoding.?);
    try testing.expectEqualSlices(Triangle, res.triangles.items, new_data.triangles.items);
}
