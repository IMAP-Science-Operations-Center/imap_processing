# Creating an Ultra packet definition file

The packet definition is made by combining the U45 and U90 telemetry definition. Galaxy page containing packet definitions: https://lasp.colorado.edu/galaxy/x/Y4AzDw).

## Modifying the Excel spreadsheets

The Excel packet definitions have some things that need to be manually updated because they
are a part of variable length packet fields. In the `MEMDUMP`, `MACRODUMP`, `MACROCHECKSUM`, `CMDTEXT`, and `CMDECHO` sheets
we need to remove all of the individual UINT definitions and change those over to one
large BYTE definition.

Original has many of these lines:

```raw
U90_MEMDUMP	DUMPDATA_000	11	8	144	UINT	NONE	DN	U90_MEMDUMP	Memory Dump 0	Memory Dump 0
U90_MEMDUMP	DUMPDATA_001	12	8	152	UINT	NONE	DN	U90_MEMDUMP	Memory Dump 1	Memory Dump 1
```

Which should be updated to one field with a BYTE dtype

```raw
U90_MEMDUMP	DUMPDATA	11	8	144	BYTE	NONE	DN	U90_MEMDUMP	Memory Dump	Memory Dump
```

## Creating the XTCE files

Run `imap_xtce TLM_U45.xls` which will produce a `TLM_U45.xml` XTCE packet definition. Do the
same for `TLM_U90.xls` to produce the `TLM_U90.xml`.

## Merging two XTCE files (U45, U90) into combined for Ultra

There are now two xml packet definitions that need to be combined together. This is done manually
by merging all non-header items together from within these XML subfields:

* ParameterTypeSet
* ParameterSet
* ContainerSet
  * Everything but the CCSDSPacket container
