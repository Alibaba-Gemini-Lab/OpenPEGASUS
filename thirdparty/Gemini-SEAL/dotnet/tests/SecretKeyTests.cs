﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using Microsoft.Research.SEAL;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;

namespace SEALNetTest
{
    [TestClass]
    public class SecretKeyTests
    {
        [TestMethod]
        public void CreateTest()
        {
            EncryptionParameters parms = new EncryptionParameters(SchemeType.BFV)
            {
                PolyModulusDegree = 64,
                PlainModulus = new Modulus(1 << 6),
                CoeffModulus = CoeffModulus.Create(64, new int[] { 40 })
            };
            SEALContext context = new SEALContext(parms,
                expandModChain: false,
                secLevel: SecLevelType.None);
            KeyGenerator keygen = new KeyGenerator(context);

            SecretKey secret = keygen.SecretKey;
            SecretKey copy = new SecretKey(secret);

            Assert.AreEqual(64ul, copy.Data.CoeffCount);
            Assert.IsTrue(copy.Data.IsNTTForm);

            SecretKey copy2 = new SecretKey();
            copy2.Set(copy);

            Assert.AreEqual(64ul, copy2.Data.CoeffCount);
            Assert.IsTrue(copy2.Data.IsNTTForm);
        }

        [TestMethod]
        public void SaveLoadTest()
        {
            EncryptionParameters parms = new EncryptionParameters(SchemeType.BFV)
            {
                PolyModulusDegree = 64,
                PlainModulus = new Modulus(1 << 6),
                CoeffModulus = CoeffModulus.Create(64, new int[] { 40 })
            };
            SEALContext context = new SEALContext(parms,
                expandModChain: false,
                secLevel: SecLevelType.None);
            KeyGenerator keygen = new KeyGenerator(context);

            SecretKey secret = keygen.SecretKey;

            Assert.AreEqual(64ul, secret.Data.CoeffCount);
            Assert.IsTrue(secret.Data.IsNTTForm);
            Assert.AreNotEqual(ParmsId.Zero, secret.ParmsId);

            SecretKey secret2 = new SecretKey();
            Assert.IsNotNull(secret2);
            Assert.AreEqual(0ul, secret2.Data.CoeffCount);
            Assert.IsFalse(secret2.Data.IsNTTForm);

            using (MemoryStream stream = new MemoryStream())
            {
                secret.Save(stream);
                stream.Seek(offset: 0, loc: SeekOrigin.Begin);
                secret2.Load(context, stream);
            }

            Assert.AreNotSame(secret, secret2);
            Assert.AreEqual(64ul, secret2.Data.CoeffCount);
            Assert.IsTrue(secret2.Data.IsNTTForm);
            Assert.AreNotEqual(ParmsId.Zero, secret2.ParmsId);
            Assert.AreEqual(secret.ParmsId, secret2.ParmsId);
        }

        [TestMethod]
        public void ExceptionsTest()
        {
            SEALContext context = GlobalContext.BFVContext;
            SecretKey key = new SecretKey();

            Utilities.AssertThrows<ArgumentNullException>(() => key = new SecretKey(null));

            Utilities.AssertThrows<ArgumentNullException>(() => key.Set(null));

            Utilities.AssertThrows<ArgumentNullException>(() => ValCheck.IsValidFor(key, null));

            Utilities.AssertThrows<ArgumentNullException>(() => key.Save(null));

            Utilities.AssertThrows<ArgumentNullException>(() => key.UnsafeLoad(null, new MemoryStream()));
            Utilities.AssertThrows<ArgumentNullException>(() => key.UnsafeLoad(context, null));
            Utilities.AssertThrows<ArgumentNullException>(() => key.Load(context, null));
            Utilities.AssertThrows<ArgumentNullException>(() => key.Load(null, new MemoryStream()));
            Utilities.AssertThrows<EndOfStreamException>(() => key.Load(context, new MemoryStream()));
        }
    }
}
