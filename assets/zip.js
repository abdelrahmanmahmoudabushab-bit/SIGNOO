// Minimal ZIP writer (store method, no compression).
//
// Recording sessions produce hundreds of files that must reach the training
// pipeline in an exact folder layout. Downloading them one by one loses that
// layout and the browser blocks the flood of downloads, so the session is packed
// into a single archive here.
//
// No compression: the payload is already-compressed video, where deflate costs CPU
// and saves nothing. Storing lets us write the archive as a straight byte stream.

const encoder = new TextEncoder();

// CRC-32 table, built once. Each ZIP entry carries a checksum of its contents.
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** Converts a JS Date to the packed MS-DOS date/time ZIP expects. */
function dosDateTime(date) {
  const time = (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
  const day = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  return { time, day };
}

/**
 * Builds a ZIP archive.
 * @param {Array<{name:string, data:Uint8Array}>} files — `name` may contain `/`.
 * @returns {Blob}
 */
export function createZip(files) {
  const chunks = [];
  const central = [];
  let offset = 0;

  const { time, day } = dosDateTime(new Date());

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const data = file.data;
    const crc = crc32(data);

    // Local file header, then the bytes themselves.
    const header = new DataView(new ArrayBuffer(30));
    header.setUint32(0, 0x04034b50, true);   // signature
    header.setUint16(4, 20, true);           // version needed
    header.setUint16(6, 0, true);            // flags
    header.setUint16(8, 0, true);            // method 0 = stored
    header.setUint16(10, time, true);
    header.setUint16(12, day, true);
    header.setUint32(14, crc, true);
    header.setUint32(18, data.length, true); // compressed size
    header.setUint32(22, data.length, true); // uncompressed size
    header.setUint16(26, nameBytes.length, true);
    header.setUint16(28, 0, true);           // extra field length

    chunks.push(new Uint8Array(header.buffer), nameBytes, data);

    // Matching central-directory record, written after all the file data.
    const entry = new DataView(new ArrayBuffer(46));
    entry.setUint32(0, 0x02014b50, true);
    entry.setUint16(4, 20, true);            // version made by
    entry.setUint16(6, 20, true);            // version needed
    entry.setUint16(8, 0, true);
    entry.setUint16(10, 0, true);
    entry.setUint16(12, time, true);
    entry.setUint16(14, day, true);
    entry.setUint32(16, crc, true);
    entry.setUint32(20, data.length, true);
    entry.setUint32(24, data.length, true);
    entry.setUint16(28, nameBytes.length, true);
    entry.setUint16(30, 0, true);            // extra
    entry.setUint16(32, 0, true);            // comment
    entry.setUint16(34, 0, true);            // disk number
    entry.setUint16(36, 0, true);            // internal attrs
    entry.setUint32(38, 0, true);            // external attrs
    entry.setUint32(42, offset, true);       // offset of local header

    central.push(new Uint8Array(entry.buffer), nameBytes);
    offset += 30 + nameBytes.length + data.length;
  }

  const centralSize = central.reduce((sum, part) => sum + part.length, 0);

  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(4, 0, true);                 // disk number
  end.setUint16(6, 0, true);                 // disk with central dir
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);           // central dir offset
  end.setUint16(20, 0, true);                // comment length

  return new Blob([...chunks, ...central, new Uint8Array(end.buffer)],
                  { type: 'application/zip' });
}
