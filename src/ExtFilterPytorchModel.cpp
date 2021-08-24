
#include <PytorchModel.hpp>
#include <VoxieClient/Array.hpp>
#include <VoxieClient/ClaimedOperation.hpp>
#include <VoxieClient/DBusClient.hpp>
#include <VoxieClient/DBusProxies.hpp>
#include <VoxieClient/DBusTypeList.hpp>
#include <VoxieClient/DBusUtil.hpp>
#include <VoxieClient/Exception.hpp>
#include <VoxieClient/Exceptions.hpp>
#include <VoxieClient/MappedBuffer.hpp>
#include <VoxieClient/QtUtil.hpp>
#include <VoxieClient/RefCountHolder.hpp>

#include <QtCore/QCommandLineParser>
#include <QtCore/QCoreApplication>
#include <QtCore/QDebug>
#include <QtCore/QString>

#include <QDebug>
#include <QtDBus/QDBusConnection>
#include <QtDBus/QDBusConnectionInterface>
#include <QtDBus/QDBusError>
#include <QtDBus/QDBusPendingReply>

int main(int argc, char* argv[]) {
  try {
    if (argc < 1)
      throw vx::Exception("de.uni_stuttgart.Voxie.PytorchModelFilter.Error",
                          "argc is smaller than 1");

    QCommandLineParser parser;
    parser.setApplicationDescription("Pytorch Model Filter");
    parser.addHelpOption();
    parser.addVersionOption();

    parser.addOptions(vx::DBusClient::options());
    parser.addOptions(vx::ClaimedOperationBase::options());

    QStringList args;
    for (char** arg = argv; *arg; arg++) args.push_back(*arg);
    int argc0 = 1;
    char* args0[2] = {argv[0], NULL};
    QCoreApplication app(argc0, args0);
    parser.process(args);

    vx::initDBusTypes();

    vx::DBusClient dbusClient(parser);

    vx::ClaimedOperation<de::uni_stuttgart::Voxie::ExternalOperationRunFilter>
        op(dbusClient,
           vx::ClaimedOperationBase::getOperationPath(parser, "RunFilter"));
    op.forwardExc([&]() {
      auto filterPath = op.op().filterObject();
      auto pars = op.op().parameters();

      auto properties = vx::dbusGetVariantValue<QMap<QString, QDBusVariant>>(
          pars[filterPath]["Properties"]);

      auto inputPath = vx::dbusGetVariantValue<QDBusObjectPath>(
          properties["de.uni_stuttgart.Voxie.Input"]);

      auto inputDataPath =
          vx::dbusGetVariantValue<QDBusObjectPath>(pars[inputPath]["Data"]);

      auto inputData = makeSharedQObject<de::uni_stuttgart::Voxie::VolumeData>(
          dbusClient.uniqueName(), inputDataPath.path(),
          dbusClient.connection());

      auto inputDataVoxel =
          makeSharedQObject<de::uni_stuttgart::Voxie::VolumeDataVoxel>(
              dbusClient.uniqueName(), inputDataPath.path(),
              dbusClient.connection());

      auto outputPath = vx::dbusGetVariantValue<QDBusObjectPath>(
          properties["de.uni_stuttgart.Voxie.Output"]);

      auto size = inputDataVoxel->arrayShape();
      QMap<QString, QDBusVariant> options;

      QSharedPointer<vx::RefObjWrapper<de::uni_stuttgart::Voxie::DataVersion>>
          volume_version;

      vx::RefObjWrapper<de::uni_stuttgart::Voxie::VolumeDataVoxel> volume(
          dbusClient, HANDLEDBUSPENDINGREPLY(dbusClient->CreateVolumeDataVoxel(
                          dbusClient.clientPath(), size, inputData->dataType(),
                          inputData->volumeOrigin(),
                          inputDataVoxel->gridSpacing(), options)));
      auto volume_data = makeSharedQObject<de::uni_stuttgart::Voxie::Data>(
          dbusClient.uniqueName(), volume.path().path(),
          dbusClient.connection());
      {
        vx::RefObjWrapper<de::uni_stuttgart::Voxie::ExternalDataUpdate> update(
            dbusClient,
            HANDLEDBUSPENDINGREPLY(volume_data->CreateUpdate(
                dbusClient.clientPath(), QMap<QString, QDBusVariant>())));


        vx::Array3<const float> inputVolume(HANDLEDBUSPENDINGREPLY(
            inputDataVoxel->GetDataReadonly(QMap<QString, QDBusVariant>())));

        vx::Array3<float> volumeData(
            HANDLEDBUSPENDINGREPLY(volume->GetDataWritable(
                update.path(), QMap<QString, QDBusVariant>())));

        PytorchModel filter;


        filter.infere(inputVolume, volumeData, op);

        volume_version = createQSharedPointer<
            vx::RefObjWrapper<de::uni_stuttgart::Voxie::DataVersion>>(
            dbusClient,
            HANDLEDBUSPENDINGREPLY(update->Finish(
                dbusClient.clientPath(), QMap<QString, QDBusVariant>())));
      }

      QMap<QString, QDBusVariant> outputResult;
      outputResult["Data"] =
          vx::dbusMakeVariant<QDBusObjectPath>(volume.path());
      outputResult["DataVersion"] =
          vx::dbusMakeVariant<QDBusObjectPath>(volume_version->path());

      QMap<QDBusObjectPath, QMap<QString, QDBusVariant>> result;
      result[outputPath] = outputResult;

      HANDLEDBUSPENDINGREPLY(
          op.op().Finish(result, QMap<QString, QDBusVariant>()));
    });
    return 0;
  } catch (vx::Exception& error) {
    QTextStream(stderr) << error.name() << ": " << error.message() << endl
                        << flush;
    return 1;
  }
}
